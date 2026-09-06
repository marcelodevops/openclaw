import { describe, expect, expectTypeOf, it } from "vitest";
import type { CodeModeTranscriptAuthority } from "../agents/code-mode-transcript-authority.js";
import type { AgentHarnessHostCapabilities } from "../agents/harness/host-capability-types.js";
import { registerTranscriptCommit } from "../agents/harness/host-private-capabilities.js";
import { commitProviderSessionTranscriptPrefix } from "./agent-harness-tool-runtime.js";

type TranscriptCommit = CodeModeTranscriptAuthority["commitPrefix"];
type ProviderCommitParams = Parameters<typeof commitProviderSessionTranscriptPrefix>[0];

describe("commitProviderSessionTranscriptPrefix", () => {
  it("requires and forwards the provider checkpoint fence", async () => {
    expectTypeOf<ProviderCommitParams["assertCurrent"]>().toEqualTypeOf<() => void>();
    expectTypeOf<
      Omit<ProviderCommitParams, "assertCurrent">
    >().not.toMatchTypeOf<ProviderCommitParams>();

    const hostCapabilities = {} as AgentHarnessHostCapabilities;
    const assertCurrent = () => undefined;
    const message = { role: "user" as const, content: "provider result" };
    let received: Parameters<TranscriptCommit> | undefined;
    registerTranscriptCommit(hostCapabilities, async (...args) => {
      received = args;
      return { kind: "replayed", results: [] };
    });

    await expect(
      commitProviderSessionTranscriptPrefix({
        assertCurrent,
        hostCapabilities,
        entries: [{ eventId: "event-1", identity: "provider:event-1", message }],
      }),
    ).resolves.toEqual({ kind: "replayed", results: [] });
    expect(received?.[0]).toEqual({
      assertCurrent,
      entries: [{ eventId: "event-1", identity: "provider:event-1", message }],
    });
    expect(received?.[1](message)).toBe(message);
  });

  it("rejects an unbound host capability", async () => {
    await expect(
      commitProviderSessionTranscriptPrefix({
        assertCurrent: () => undefined,
        hostCapabilities: {} as AgentHarnessHostCapabilities,
        entries: [],
      }),
    ).rejects.toThrow("provider transcript commit requires host transcript capability");
  });
});
