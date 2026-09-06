import path from "node:path";
import { afterEach, expect, it, vi } from "vitest";
import {
  appendTranscriptMessageSync,
  loadTranscriptEventsSync,
  replaceSessionEntrySync,
} from "../config/sessions/session-accessor.js";
import {
  resolveSqliteTranscriptScope,
  toDatabaseOptions,
} from "../config/sessions/session-accessor.sqlite-scope.js";
import {
  closeOpenClawAgentDatabasesForTest,
  openOpenClawAgentDatabase,
} from "../state/openclaw-agent-db.js";
import {
  withOpenClawTestState,
  type OpenClawTestState,
} from "../test-utils/openclaw-test-state.js";
import {
  bindCodeModeTranscriptAuthority,
  CodeModeTranscriptAuthority,
  resolveCodeModeTranscriptAuthority,
} from "./code-mode-transcript-authority.js";
import { resetCodeModeTestState, testing } from "./code-mode.test-support.js";
import { guardSessionManager } from "./session-tool-result-guard-wrapper.js";
import { SessionManager } from "./sessions/session-manager.js";

afterEach(() => {
  closeOpenClawAgentDatabasesForTest();
  resetCodeModeTestState();
});

function target(state: OpenClawTestState) {
  const scope = {
    agentId: "main",
    env: state.env,
    expectedWriterRunId: "writer",
    sessionId: "authority-session",
    sessionKey: "agent:main:authority",
    storePath: path.join(state.sessionsDir(), "sessions.json"),
  };
  replaceSessionEntrySync(scope, {
    activeWriterRunId: scope.expectedWriterRunId,
    sessionId: scope.sessionId,
    updatedAt: 1,
  });
  return scope;
}

const prefix = {
  entries: [
    {
      eventId: "provider-event",
      identity: "provider:event",
      message: { role: "user" as const, content: "source" },
    },
  ],
};

function appendUnkeyedBase(scope: ReturnType<typeof target>) {
  const result = appendTranscriptMessageSync(scope, {
    eventId: "base-event",
    message: { role: "user" as const, content: "base" },
  });
  if (!result.ok || !result.value?.anchor) {
    throw new Error("failed to append transcript test base");
  }
  return result.value.anchor;
}

function readRawTranscriptState(scope: ReturnType<typeof target>) {
  const database = openOpenClawAgentDatabase(
    toDatabaseOptions(resolveSqliteTranscriptScope(scope)),
  );
  return {
    database,
    identities: database.db
      .prepare(
        `SELECT event_id, parent_id, message_idempotency_key, seq
         FROM transcript_event_identities
         WHERE session_id = ?
         ORDER BY seq`,
      )
      .all(scope.sessionId),
    rows: database.db
      .prepare(
        `SELECT event_json, seq
         FROM transcript_events
         WHERE session_id = ?
         ORDER BY seq`,
      )
      .all(scope.sessionId),
  };
}

it("commits a rewritten source once and replays it without rerunning the hook", async () => {
  await withOpenClawTestState({ label: "authority-replay" }, async (state) => {
    const authority = new CodeModeTranscriptAuthority(target(state));
    const prepare = vi.fn(() => ({ role: "user" as const, content: "rewritten" }));
    await expect(authority.commitPrefix(prefix, prepare)).resolves.toMatchObject({
      kind: "committed",
    });
    expect(prepare).toHaveBeenCalledOnce();
    const replayPrepare = vi.fn(() => ({ role: "user" as const, content: "wrong" }));
    const replay = await authority.commitPrefix(prefix, replayPrepare);
    expect(replay).toMatchObject({ kind: "replayed" });
    expect(replayPrepare).not.toHaveBeenCalled();
    expect(replay.results).toHaveLength(1);
    expect(replay.results[0]?.identity).toBe("provider:event");
    expect(replay.results[0]?.message).toMatchObject({
      content: "rewritten",
      idempotencyKey: "provider:event",
      __openclaw: { providerSourceFingerprint: expect.stringMatching(/^[a-f0-9]{32}$/u) },
    });
    const emptyPrepare = vi.fn();
    await expect(authority.commitPrefix({ entries: [] }, emptyPrepare)).resolves.toMatchObject({
      kind: "replayed",
      results: [],
    });
    expect(emptyPrepare).not.toHaveBeenCalled();
  });
});

it("replays from raw topology after a cold reopen without repairing a dirty projection", async () => {
  await withOpenClawTestState({ label: "authority-dirty-replay" }, async (state) => {
    const scope = target(state);
    const authority = new CodeModeTranscriptAuthority(scope);
    await expect(
      authority.commitPrefix(prefix, () => ({ role: "user", content: "rewritten" })),
    ).resolves.toMatchObject({ kind: "committed" });
    const before = readRawTranscriptState(scope);
    before.database.db
      .prepare("UPDATE session_transcript_index_state SET needs_rebuild = 1 WHERE session_id = ?")
      .run(scope.sessionId);
    const rows = before.rows;
    const identities = before.identities;
    closeOpenClawAgentDatabasesForTest();

    const prepare = vi.fn(() => ({ role: "user" as const, content: "wrong" }));
    await expect(
      new CodeModeTranscriptAuthority(scope).commitPrefix(prefix, prepare),
    ).resolves.toMatchObject({
      kind: "replayed",
    });
    expect(prepare).not.toHaveBeenCalled();

    const after = readRawTranscriptState(scope);
    expect(after.rows).toEqual(rows);
    expect(after.identities).toEqual(identities);
    expect(
      after.database.db
        .prepare("SELECT needs_rebuild FROM session_transcript_index_state WHERE session_id = ?")
        .get(scope.sessionId),
    ).toEqual({ needs_rebuild: 1 });
  });
});

it("rejects a malformed SQLite message instead of replaying it", async () => {
  await withOpenClawTestState({ label: "authority-malformed-replay" }, async (state) => {
    const scope = target(state);
    const proof = "0".repeat(32);
    appendTranscriptMessageSync(scope, {
      eventId: "malformed-event",
      message: {
        role: "user",
        idempotencyKey: "provider:malformed",
        __openclaw: { providerSourceFingerprint: proof },
      } as never,
    });
    const authority = new CodeModeTranscriptAuthority(scope);
    await expect(
      authority.commitPrefix(
        {
          entries: [
            {
              eventId: "malformed-event",
              identity: "provider:malformed",
              message: { role: "user", content: "source" },
              sourceFingerprint: proof,
            },
          ],
        },
        vi.fn(),
      ),
    ).resolves.toEqual({ kind: "conflict", reason: "prefix-mismatch" });
  });
});

it("validates every unkeyed base anchor field for empty commits", async () => {
  await withOpenClawTestState({ label: "authority-base-anchor" }, async (state) => {
    const scope = target(state);
    const authority = new CodeModeTranscriptAuthority(scope);
    const baseAnchor = appendUnkeyedBase(scope);
    expect(baseAnchor.idempotencyKey).toBeUndefined();
    await expect(
      authority.commitPrefix({ baseAnchor, entries: [] }, vi.fn()),
    ).resolves.toMatchObject({ kind: "replayed", results: [] });
    await expect(
      authority.commitPrefix(
        { baseAnchor: { ...baseAnchor, generation: "stale-generation" }, entries: [] },
        vi.fn(),
      ),
    ).resolves.toEqual({ kind: "conflict", reason: "base-anchor-mismatch" });
    await expect(
      authority.commitPrefix(
        { baseAnchor: { ...baseAnchor, rawSeq: baseAnchor.rawSeq + 1 }, entries: [] },
        vi.fn(),
      ),
    ).resolves.toEqual({ kind: "conflict", reason: "base-anchor-mismatch" });
  });
});

it("rejects a concurrent branch change inside the canonical write transaction", async () => {
  await withOpenClawTestState({ label: "authority-race" }, async (state) => {
    const scope = target(state);
    const authority = new CodeModeTranscriptAuthority(scope);
    const baseAnchor = appendUnkeyedBase(scope);
    const result = await authority.commitPrefix(
      { baseAnchor, entries: prefix.entries },
      (message) => {
        appendTranscriptMessageSync(scope, {
          eventId: "racing-event",
          message: { role: "user", content: "racing write" },
        });
        return message;
      },
    );
    expect(result).toEqual({ kind: "conflict", reason: "transaction-drift" });
    const messages = loadTranscriptEventsSync(scope).flatMap((event) =>
      event.type === "message" ? [event.message] : [],
    );
    expect(messages).toEqual([
      expect.objectContaining({ content: "base" }),
      expect.objectContaining({ content: "racing write" }),
    ]);
  });
});

it("rolls back when close revokes authority after hook preparation", async () => {
  await withOpenClawTestState({ label: "authority-close" }, async (state) => {
    const scope = target(state);
    const authority = new CodeModeTranscriptAuthority(scope);
    await expect(
      authority.commitPrefix(prefix, (message) => {
        authority.close();
        return message;
      }),
    ).rejects.toThrow("authority is closed");
    expect(loadTranscriptEventsSync(scope).filter((event) => event.type === "message")).toEqual([]);
  });
});

it("certifies waiting only after the guarded SessionManager makes it durable", async () => {
  await withOpenClawTestState({ label: "authority-session-manager" }, async (state) => {
    const scope = target(state);
    const authority = new CodeModeTranscriptAuthority(scope);
    const attempt = {};
    bindCodeModeTranscriptAuthority(attempt, authority);
    const unguarded = SessionManager.open(scope, state.workspaceDir);
    const attemptAuthority = resolveCodeModeTranscriptAuthority(attempt);
    if (!attemptAuthority) {
      throw new Error("attempt did not expose its transcript authority");
    }
    bindCodeModeTranscriptAuthority(unguarded, attemptAuthority);
    const manager = guardSessionManager(unguarded, {
      runId: "cm-test",
      allowedToolNames: ["exec"],
    });
    const reserve = authority.reserve.bind(authority);
    const certificationChecks: string[] = [];
    const reserveSpy = vi.spyOn(authority, "reserve").mockImplementation((message) => {
      const reservation = reserve(message);
      if (!reservation) {
        return undefined;
      }
      return {
        ...reservation,
        commit: () => {
          const durable = loadTranscriptEventsSync(scope).find(
            (event) =>
              event.type === "message" &&
              event.message.role === "toolResult" &&
              event.message.idempotencyKey === reservation.identity,
          );
          expect(durable).toBeDefined();
          certificationChecks.push(reservation.identity);
          reservation.commit();
        },
      };
    });
    const identities: string[] = [];
    for (const replayId of ["replay-a", "replay-b"]) {
      manager.appendMessage({
        role: "assistant",
        content: [{ type: "toolCall", id: "call-test", name: "exec", arguments: {} }],
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
        timestamp: 1,
      });
      testing.activeRuns.set("cm-test", {
        expiresAt: Date.now() + 60_000,
        parentToolCallId: "call-test",
        replayId,
      } as never);
      const message = {
        role: "toolResult" as const,
        toolCallId: "call-test",
        toolName: "exec",
        content: [{ type: "text" as const, text: replayId }],
        isError: false,
      };
      authority.captureWaiting({ runId: "cm-test", toolCallId: "call-test", toolName: "exec" });
      manager.appendMessage(message);
      expect(authority.verifyWaiting("cm-test")).toBe(true);
      identities.push(certificationChecks.at(-1)!);
    }
    expect(reserveSpy).toHaveBeenCalledTimes(2);
    expect(certificationChecks).toEqual(identities);
    expect(identities).toEqual([
      expect.stringMatching(/^code-mode-result:[a-f0-9]{32}$/u),
      expect.stringMatching(/^code-mode-result:[a-f0-9]{32}$/u),
    ]);
    expect(identities[1]).not.toBe(identities[0]);
    closeOpenClawAgentDatabasesForTest();
    expect(
      SessionManager.open(scope, state.workspaceDir)
        .buildSessionContext()
        .messages.filter((message) => message.role === "toolResult"),
    ).toMatchObject([
      { toolCallId: "call-test", content: [{ text: "replay-a" }] },
      { toolCallId: "call-test", content: [{ text: "replay-b" }] },
    ]);
  });
});
