import type { AgentMessage } from "openclaw/plugin-sdk/agent-harness-runtime";
import { upsertSessionEntry } from "openclaw/plugin-sdk/session-store-runtime";
import {
  appendSessionTranscriptMessagesByIdentity,
  readSessionTranscriptEvents,
} from "openclaw/plugin-sdk/session-transcript-runtime";
import { closeOpenClawAgentDatabasesForTest } from "openclaw/plugin-sdk/sqlite-runtime-testing";
import { asOptionalRecord } from "openclaw/plugin-sdk/string-coerce-runtime";
import { withTempDir } from "openclaw/plugin-sdk/test-env";
import { afterEach, expect, it, vi } from "vitest";
import { captureCodexSettledTurnFinalizationContext } from "./settled-turn-context.js";
import { CodexTranscriptCheckpoint } from "./transcript-checkpoint.js";
import { codexTranscriptMirrorRuntime } from "./transcript-mirror.js";
import { attachCodexMirrorIdentity } from "./upstream-prompt-provenance.js";

const commitPrefix = vi.hoisted(() => vi.fn());

vi.mock("openclaw/plugin-sdk/agent-harness-tool-runtime", () => ({
  commitProviderSessionTranscriptPrefix: commitPrefix,
}));

afterEach(() => {
  vi.restoreAllMocks();
  commitPrefix.mockReset();
  closeOpenClawAgentDatabasesForTest();
});

function assistantMessage(): AgentMessage {
  return {
    role: "assistant",
    content: [{ type: "toolCall", id: "call-1", name: "exec", arguments: {} }],
    api: "openai-responses",
    provider: "openai",
    model: "gpt-test",
    usage: {
      input: 0,
      output: 0,
      cacheRead: 0,
      cacheWrite: 0,
      totalTokens: 0,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
    },
    stopReason: "toolUse",
    timestamp: 2,
  };
}

async function waitingFixture(tempDir: string) {
  const target = {
    agentId: "main",
    sessionId: "session-cold",
    sessionKey: "agent:main:session-cold",
    storePath: `${tempDir}/sessions.json`,
  };
  await upsertSessionEntry({
    ...target,
    entry: { sessionId: target.sessionId, updatedAt: 1 },
  });
  const hostCapabilities = {
    kind: "agent-harness-host-capability",
    version: 1,
    assertActive: () => undefined,
  } as never;
  commitPrefix.mockImplementation(
    async ({
      baseAnchor,
      entries,
    }: {
      baseAnchor?: { entryId: string };
      entries: Array<{ eventId: string; identity: string; message: AgentMessage }>;
    }) => {
      const results = await appendSessionTranscriptMessagesByIdentity({
        ...target,
        messages: entries.map((entry, index) => ({
          eventId: entry.eventId,
          idempotencyLookup: "scan" as const,
          message: {
            ...entry.message,
            __openclaw: {
              ...asOptionalRecord(Reflect.get(entry.message, "__openclaw")),
              providerSourceFingerprint: "0".repeat(32),
            },
          },
          parentId: index ? entries[index - 1]!.eventId : baseAnchor?.entryId,
        })),
      });
      return {
        kind: "committed",
        results: results.map((result, index) => ({
          anchor: result.anchor,
          identity: entries[index]!.identity,
          message: result.message,
        })),
      };
    },
  );
  const messages = [
    attachCodexMirrorIdentity({ role: "user", content: "pause", timestamp: 1 }, "turn-1:prompt"),
    attachCodexMirrorIdentity(assistantMessage(), "turn-1:tool:call-1:call"),
    attachCodexMirrorIdentity(
      {
        role: "toolResult",
        toolCallId: "call-1",
        toolName: "exec",
        content: [{ type: "text", text: "waiting" }],
        details: { status: "waiting" },
        isError: false,
        timestamp: 3,
      },
      "turn-1:tool:call-1:result",
    ),
  ];
  const checkpoint = new CodexTranscriptCheckpoint(
    {
      hostCapabilities,
      runId: "run-1",
      sessionId: target.sessionId,
      sessionTarget: target,
    } as never,
    "thread-1",
    "turn-1",
  );
  return { checkpoint, hostCapabilities, messages, target };
}

async function persistWaitingTurn(tempDir: string) {
  const { checkpoint, hostCapabilities, messages, target } = await waitingFixture(tempDir);
  messages.forEach((message, index) =>
    checkpoint.enqueue({ read: () => message, requiredCommit: index === messages.length - 1 }),
  );
  await checkpoint.flush();
  expect(commitPrefix).toHaveBeenCalledOnce();

  closeOpenClawAgentDatabasesForTest();
  const mirrorOutcome = await codexTranscriptMirrorRuntime.mirrorBestEffort({
    params: {
      hostCapabilities,
      runId: "run-final",
      sessionId: target.sessionId,
      sessionKey: target.sessionKey,
      sessionTarget: target,
      suppressNextUserMessagePersistence: true,
    } as never,
    result: { messagesSnapshot: messages } as never,
    agentId: target.agentId,
    sessionKey: target.sessionKey,
    notifyUserMessagePersisted: () => undefined,
    cwd: tempDir,
    threadId: "thread-1",
    turnId: "turn-1",
  });
  closeOpenClawAgentDatabasesForTest();
  return { messages, mirrorOutcome, target };
}

it("replays one attested waiting result after a cold reopen", async () => {
  await withTempDir("codex-waiting-cold-", async (tempDir) => {
    const { mirrorOutcome, target } = await persistWaitingTurn(tempDir);
    expect(mirrorOutcome.mirroredMessages).toHaveLength(3);
    const toolResults = (await readSessionTranscriptEvents(target)).filter(
      (event) => event.type === "message" && event.message?.role === "toolResult",
    );
    expect(toolResults).toHaveLength(1);
    expect(toolResults[0]).toMatchObject({
      id: "codex-app-server:thread-1:turn-1:tool:call-1:result",
      message: {
        __openclaw: {
          mirrorIdentity: "turn-1:tool:call-1:result",
          mirrorOrigin: "codex-app-server",
          mirrorSourceFingerprint: expect.stringMatching(/^[a-f0-9]{32}$/u),
          providerSourceFingerprint: expect.stringMatching(/^[a-f0-9]{32}$/u),
        },
      },
    });
  });
});

it("commits only the missing suffix after a partial ordinary mirror failure", async () => {
  await withTempDir("codex-waiting-prefix-", async (tempDir) => {
    const { checkpoint, messages, target } = await waitingFixture(tempDir);
    const sources = [...messages];
    sources.slice(0, 2).forEach((_, index) => checkpoint.enqueue({ read: () => sources[index] }));
    const mirror = codexTranscriptMirrorRuntime.mirror;
    vi.spyOn(codexTranscriptMirrorRuntime, "mirror").mockImplementationOnce(async (params) => {
      await mirror({ ...params, messages: params.messages.slice(0, 1) });
      throw new Error("partial mirror failure");
    });
    await checkpoint.flush();

    checkpoint.enqueue({ read: () => sources[2], requiredCommit: true });
    await checkpoint.flush();
    expect(commitPrefix).toHaveBeenCalledOnce();

    const persisted = (await readSessionTranscriptEvents(target)).filter(
      (event) => event.type === "message",
    );
    expect(persisted.map((event) => event.message.role)).toEqual([
      "user",
      "assistant",
      "toolResult",
    ]);
    expect(new Set(persisted.map((event) => event.message.idempotencyKey)).size).toBe(3);
  });
});

it("rejects a changed mirrored prefix without committing its suffix", async () => {
  await withTempDir("codex-waiting-prefix-drift-", async (tempDir) => {
    const { checkpoint, messages, target } = await waitingFixture(tempDir);
    const sources = [...messages];
    sources.slice(0, 2).forEach((_, index) => checkpoint.enqueue({ read: () => sources[index] }));
    const mirror = codexTranscriptMirrorRuntime.mirror;
    vi.spyOn(codexTranscriptMirrorRuntime, "mirror").mockImplementationOnce(async (params) => {
      await mirror({ ...params, messages: params.messages.slice(0, 1) });
      throw new Error("partial mirror failure");
    });
    await checkpoint.flush();
    sources[0] = attachCodexMirrorIdentity(
      { role: "user", content: "changed pause", timestamp: 1 },
      "turn-1:prompt",
    );
    checkpoint.enqueue({ read: () => sources[2], requiredCommit: true });

    await expect(checkpoint.flush()).rejects.toThrow("prefix drifted");
    expect(commitPrefix).not.toHaveBeenCalled();
    const persisted = (await readSessionTranscriptEvents(target)).filter(
      (event) => event.type === "message",
    );
    expect(persisted).toHaveLength(1);
    expect(persisted[0]?.message.content).toBe("pause");
  });
});

it("accepts identical cold settled evidence and rejects payload drift", async () => {
  await withTempDir("codex-waiting-settled-", async (tempDir) => {
    const { messages, mirrorOutcome, target } = await persistWaitingTurn(tempDir);
    const capture = (mirroredMessages: AgentMessage[], settledMessages: AgentMessage[]) =>
      captureCodexSettledTurnFinalizationContext({
        ...target,
        sessionFile: target.storePath,
        sessionTarget: target,
        mirroredMessages,
        settledMessages,
        turnId: "turn-1",
        model: "gpt-test",
      });
    await expect(capture(mirrorOutcome.mirroredMessages, messages)).resolves.toMatchObject({
      source: "harness",
    });

    const mismatched = structuredClone(mirrorOutcome.mirroredMessages);
    const result = mismatched.at(-1);
    if (!result || result.role !== "toolResult") {
      throw new Error("expected mirrored waiting result");
    }
    result.content = [{ type: "text", text: "different result" }];
    await expect(capture(mismatched, mismatched)).resolves.toBeUndefined();
  });
});
