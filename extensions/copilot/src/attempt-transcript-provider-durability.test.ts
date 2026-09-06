import type { AgentMessage } from "openclaw/plugin-sdk/agent-harness-runtime";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  cleanupAttemptTranscriptJournalFixtures,
  createFixture,
  event,
} from "./attempt-transcript-journal.test-helpers.js";
import { buildSuspendableToolResultMessage } from "./event-bridge-transcript.js";

const commitPrefix = vi.hoisted(() => vi.fn());

vi.mock("openclaw/plugin-sdk/agent-harness-tool-runtime", () => ({
  commitProviderSessionTranscriptPrefix: commitPrefix,
}));

afterEach(async () => {
  commitPrefix.mockReset();
  await cleanupAttemptTranscriptJournalFixtures();
});

describe("Copilot provider transcript durability", () => {
  it("commits one canonical assistant/result group for an outer waiting result", async () => {
    commitPrefix.mockImplementation(
      async ({ entries }: { entries: Array<{ message: AgentMessage }> }) => ({
        kind: "committed",
        results: entries.map((entry, index) => ({
          anchor: { entryId: `event-${index}`, idempotencyKey: `key-${index}`, messageSeq: index },
          message: entry.message,
        })),
      }),
    );
    const { attempt, journal, session } = await createFixture();
    await journal.persistInitialUser();
    session.emit(event("user.message", "initial-user", { content: "pause" }));
    session.emit(
      event("assistant.message", "assistant-waiting", {
        content: "",
        messageId: "assistant-waiting",
        toolRequests: [{ arguments: {}, name: "exec", toolCallId: "exec-waiting-1" }],
      }),
    );

    const waitingMessage = buildSuspendableToolResultMessage({
      providerResult: { resultType: "success", textResultForLlm: "waiting" },
      result: {
        content: [{ type: "text", text: "outer result" }],
        details: { runId: "run-waiting-1", status: "waiting" },
      },
      startedAt: 2,
      toolCallId: "exec-waiting-1",
      toolName: "exec",
    });
    expect(waitingMessage.details).toEqual({
      runId: "run-waiting-1",
      status: "waiting",
    });
    expect(waitingMessage.details).not.toHaveProperty("details");
    expect(
      buildSuspendableToolResultMessage({
        providerResult: { resultType: "success", textResultForLlm: "done" },
        result: { content: [{ type: "text", text: "done" }] },
        startedAt: 2,
        toolCallId: "exec-complete-1",
        toolName: "exec",
      }),
    ).not.toHaveProperty("details");
    const receipt = journal.recordProviderToolResult(waitingMessage);
    session.emit(
      event("tool.execution_complete", "sdk-duplicate", {
        result: { content: "waiting" },
        success: true,
        toolCallId: "exec-waiting-1",
      }),
    );

    expect(commitPrefix).not.toHaveBeenCalled();
    await journal.barrier("waiting result");
    await expect(receipt).resolves.toBeUndefined();
    expect(commitPrefix).toHaveBeenCalledOnce();
    expect(commitPrefix.mock.calls[0]?.[0].assertCurrent).toBe(
      attempt.hostCapabilities!.assertActive,
    );
    expect(
      commitPrefix.mock.calls[0]?.[0].entries.map(
        (entry: { message: AgentMessage }) => entry.message.role,
      ),
    ).toEqual(["assistant", "toolResult"]);
    const committedResult = commitPrefix.mock.calls[0]?.[0].entries.at(-1)?.message;
    expect(committedResult).toMatchObject({
      details: { runId: "run-waiting-1", status: "waiting" },
      role: "toolResult",
    });
    expect(committedResult?.details).not.toHaveProperty("details");
    expect(journal.snapshot().messagesSnapshot.map((message) => message.role)).toEqual([
      "user",
      "assistant",
      "toolResult",
    ]);
  });

  it("does not deadlock a mixed waiting and computer result group", async () => {
    commitPrefix.mockImplementation(
      async ({ entries }: { entries: Array<{ message: AgentMessage }> }) => ({
        kind: "committed",
        results: entries.map((entry, index) => ({
          anchor: { entryId: `event-${index}` },
          message: entry.message,
        })),
      }),
    );
    const { journal, session } = await createFixture();
    await journal.persistInitialUser();
    session.emit(event("user.message", "initial-user", { content: "pause" }));
    session.emit(
      event("assistant.message", "assistant-mixed", {
        content: "",
        messageId: "assistant-mixed",
        toolRequests: [
          { arguments: {}, name: "exec", toolCallId: "exec-waiting" },
          { arguments: {}, name: "computer", toolCallId: "computer-1" },
        ],
      }),
    );
    const receipt = journal.recordProviderToolResult({
      role: "toolResult",
      toolCallId: "exec-waiting",
      toolName: "exec",
      content: [{ type: "text", text: "waiting" }],
      details: { status: "waiting" },
      isError: false,
    });
    session.emit(
      event("tool.execution_complete", "computer-result", {
        result: { content: "frame" },
        success: true,
        toolCallId: "computer-1",
      }),
    );
    await journal.barrier("mixed group");
    await expect(receipt).resolves.toBeUndefined();

    expect(commitPrefix).toHaveBeenCalledOnce();
    expect(
      commitPrefix.mock.calls[0]?.[0].entries.map(
        (entry: { message: AgentMessage }) => entry.message.role,
      ),
    ).toEqual(["assistant", "toolResult", "toolResult"]);
  });

  it("rejects a malformed provider replay message", async () => {
    commitPrefix.mockResolvedValue({
      kind: "replayed",
      results: [{ anchor: { entryId: "malformed" }, message: { role: "toolResult" } }],
    });
    const { journal, session } = await createFixture();
    await journal.persistInitialUser();
    session.emit(event("user.message", "initial-user", { content: "pause" }));
    session.emit(
      event("assistant.message", "assistant-malformed", {
        content: "",
        messageId: "assistant-malformed",
        toolRequests: [{ arguments: {}, name: "exec", toolCallId: "exec-malformed" }],
      }),
    );
    const receipt = journal
      .recordProviderToolResult({
        role: "toolResult",
        toolCallId: "exec-malformed",
        toolName: "exec",
        content: [{ type: "text", text: "waiting" }],
        details: { status: "waiting" },
        isError: false,
      })
      .catch((error: unknown) => error);

    await expect(journal.barrier("malformed replay")).rejects.toThrow(
      "replayed an invalid message",
    );
    await expect(receipt).resolves.toBeInstanceOf(Error);
  });

  it("rejects every late provider receipt with the original commit failure", async () => {
    const failure = new Error("injected provider commit failure");
    commitPrefix.mockRejectedValue(failure);
    const { journal, session } = await createFixture();
    await journal.persistInitialUser();
    session.emit(event("user.message", "initial-user", { content: "pause" }));
    session.emit(
      event("assistant.message", "assistant-failed", {
        content: "",
        messageId: "assistant-failed",
        toolRequests: [{ arguments: {}, name: "exec", toolCallId: "exec-failed" }],
      }),
    );
    const receipt = journal.recordProviderToolResult({
      role: "toolResult",
      toolCallId: "exec-failed",
      toolName: "exec",
      content: [{ type: "text", text: "waiting" }],
      details: { status: "waiting" },
      isError: false,
    });
    const rejectedReceipt = receipt.catch((error: unknown) => error);

    await expect(journal.barrier("final response")).rejects.toThrow(
      "injected provider commit failure",
    );
    await expect(rejectedReceipt).resolves.toBe(failure);
    const lateMessage = {
      role: "toolResult" as const,
      toolCallId: "exec-failed",
      toolName: "exec",
      content: [{ type: "text" as const, text: "late waiting" }],
      details: { status: "waiting" },
      isError: false,
    };
    await expect(
      Promise.all([
        journal.recordProviderToolResult(lateMessage).catch((error: unknown) => error),
        journal.recordProviderToolResult(lateMessage).catch((error: unknown) => error),
      ]),
    ).resolves.toEqual([failure, failure]);
    expect(commitPrefix).toHaveBeenCalledOnce();
  });
});
