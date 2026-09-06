import {
  embeddedAgentLog,
  formatErrorMessage,
  projectAgentHarnessTranscriptMessageForDisplay,
  type AgentMessage,
  type EmbeddedRunAttemptParamsV2,
} from "openclaw/plugin-sdk/agent-harness-runtime";
import { commitProviderSessionTranscriptPrefix } from "openclaw/plugin-sdk/agent-harness-tool-runtime";
import * as mirrorAttestation from "./transcript-mirror-attestation.js";
import { codexTranscriptMirrorRuntime } from "./transcript-mirror.js";
import { attachCodexMirrorIdentity, readMirrorIdentity } from "./upstream-prompt-provenance.js";

export type CodexTranscriptCheckpointEntry = {
  read: () => AgentMessage | undefined;
  ready?: () => boolean;
  requiredCommit?: boolean;
};

function providerEntry(threadId: string, message: AgentMessage) {
  const mirrorIdentity = readMirrorIdentity(message);
  if (!mirrorIdentity) {
    throw new Error("Codex provider transcript entry lacks a mirror identity");
  }
  if (message.role !== "user" && message.role !== "assistant" && message.role !== "toolResult") {
    throw new Error("Codex provider transcript entry has an unsupported role");
  }
  const identity = `codex-app-server:${threadId}:${mirrorIdentity}`;
  const attested = mirrorAttestation.attachCodexMirrorAttestation(
    message,
    mirrorAttestation.fingerprintCodexMirrorSourceMessage(message),
  );
  return { eventId: identity, identity, message: { ...attested, idempotencyKey: identity } };
}

/** Commits completed work in receipt order without waiting for the enclosing turn. */
export class CodexTranscriptCheckpoint {
  private readonly pending: CodexTranscriptCheckpointEntry[] = [];
  private readonly commentaryItemIds = new Set<string>();
  private lastTimestamp = 0;
  private writing = Promise.resolve();
  private tainted = false;
  private closed = false;
  private abandoned = false;

  constructor(
    private readonly params: EmbeddedRunAttemptParamsV2,
    private readonly threadId: string,
    private readonly turnId: string,
  ) {}

  nextTimestamp = (): number => {
    // Commentary and tool mirrors share this clock so equal wall-clock values
    // still preserve the app-server receipt order in the durable transcript.
    this.lastTimestamp = Math.max(Date.now(), this.lastTimestamp + 1);
    return this.lastTimestamp;
  };

  enqueueCommentary = (itemId: string, entry: CodexTranscriptCheckpointEntry): void => {
    if (
      this.params.config?.ui?.prefs?.chatPersistCommentary === false ||
      this.commentaryItemIds.has(itemId)
    ) {
      return;
    }
    this.commentaryItemIds.add(itemId);
    this.enqueue({
      ...entry,
      read: () => {
        const message = entry.read();
        return message
          ? attachCodexMirrorIdentity(message, `${this.turnId}:commentary:${itemId}`)
          : undefined;
      },
    });
  };

  enqueue = (entry: CodexTranscriptCheckpointEntry): void => {
    if (this.params.sessionTarget && !this.closed) {
      this.pending.push(entry);
    }
  };

  abandon(): void {
    this.closed = this.abandoned = true;
    this.pending.length = 0;
  }

  flush(close = false): Promise<void> {
    if (this.closed) {
      return this.writing;
    }
    this.closed = close;
    this.writing = this.writing.then(async () => {
      const assertWriteCurrent = () => {
        if (this.abandoned) {
          throw new Error("Codex transcript checkpoint was retired before write");
        }
      };
      assertWriteCurrent();
      // An unfinished commentary item or linked raw patch output owns its place
      // in history. Later work cannot overtake it; teardown records what arrived.
      const blocked = close ? -1 : this.pending.findIndex((entry) => entry.ready?.() === false);
      const count = blocked < 0 ? this.pending.length : blocked;
      if (count === 0) {
        return;
      }
      const taint = { tainted: this.tainted };
      const pending = this.pending.slice(0, count);
      const requiredIndex = pending.findIndex((entry) => entry.requiredCommit);
      const messages = pending.flatMap((entry) => {
        const message = entry.read();
        return message
          ? [
              projectAgentHarnessTranscriptMessageForDisplay({
                hidden: this.params.trigger === "memory",
                message: mirrorAttestation.applyCodexTranscriptTaint(message, taint),
              }),
            ]
          : [];
      });
      assertWriteCurrent();
      try {
        if (requiredIndex >= 0) {
          const prefix = await codexTranscriptMirrorRuntime.mirror({
            assertWriteCurrent,
            ...this.params.sessionTarget,
            sessionId: this.params.sessionId,
            cwd: this.params.workspaceDir,
            messages: messages.slice(0, requiredIndex),
            idempotencyScope: `codex-app-server:${this.threadId}`,
            inspectOnly: true,
          });
          const last = prefix.messagesPresent.at(-1);
          const identity = last ? readMirrorIdentity(last) : undefined;
          const outcome = await commitProviderSessionTranscriptPrefix({
            assertCurrent: assertWriteCurrent,
            hostCapabilities: this.params.hostCapabilities,
            ...(identity ? { baseAnchor: prefix.anchorsByMirrorIdentity.get(identity) } : {}),
            entries: messages
              .slice(prefix.messagesPresent.length)
              .map((message) => providerEntry(this.threadId, message)),
          });
          if (outcome.kind !== "committed" && outcome.kind !== "replayed") {
            throw new Error(`Codex provider transcript commit ${outcome.kind}`);
          }
        } else {
          await codexTranscriptMirrorRuntime.mirror({
            assertWriteCurrent,
            ...this.params.sessionTarget,
            sessionId: this.params.sessionId,
            cwd: this.params.workspaceDir,
            messages,
            idempotencyScope: `codex-app-server:${this.threadId}`,
            runId: this.params.runId,
            runMirrorIdentityPrefix: `${this.turnId}:`,
            config: this.params.config,
          });
        }
        this.pending.splice(0, count);
        this.tainted = taint.tainted;
      } catch (error) {
        if (requiredIndex >= 0) {
          throw error;
        }
        // Keep the pending prefix; persistence failure must not retry completed tool work.
        embeddedAgentLog.warn("failed to checkpoint codex app-server transcript", {
          runId: this.params.runId,
          sessionId: this.params.sessionId,
          error: formatErrorMessage(error),
        });
      }
    });
    return this.writing;
  }
}
