import Supermemory from "supermemory";
import { CONFIG, SUPERMEMORY_API_KEY, isConfigured } from "../config.js";
import { log } from "./logger.js";
import type {
  ConversationIngestResponse,
  ConversationMessage,
  MemoryType,
} from "../types/index.js";

const TIMEOUT_MS = 30000;
const MAX_CONVERSATION_CHARS = 100_000;

function withTimeout<T>(promise: Promise<T>, ms: number): Promise<T> {
  return Promise.race([
    promise,
    new Promise<T>((_, reject) =>
      setTimeout(() => reject(new Error(`Timeout after ${ms}ms`)), ms)
    ),
  ]);
}

export class SupermemoryClient {
  private client: Supermemory | null = null;

  private formatConversationMessage(message: ConversationMessage): string {
    const content =
      typeof message.content === "string"
        ? message.content
        : message.content
            .map((part) =>
              part.type === "text"
                ? part.text
                : `[image] ${part.imageUrl.url}`
            )
            .join("\n");

    const trimmed = content.trim();
    if (trimmed.length === 0) {
      return `[${message.role}]`;
    }
    return `[${message.role}] ${trimmed}`;
  }

  private formatConversationTranscript(messages: ConversationMessage[]): string {
    return messages
      .map((message, idx) => `${idx + 1}. ${this.formatConversationMessage(message)}`)
      .join("\n");
  }

  private getClient(): Supermemory {
    if (!this.client) {
      if (!isConfigured()) {
        throw new Error("SUPERMEMORY_API_KEY not set");
      }
      this.client = new Supermemory({ apiKey: SUPERMEMORY_API_KEY });
      this.client.settings.update({
	     	shouldLLMFilter: true,
	      filterPrompt: CONFIG.filterPrompt
      })
    }
    return this.client;
  }

  async searchMemories(query: string, containerTag: string) {
    log("searchMemories: start", { containerTag });
    try {
      const searchParams = {
        q: query,
        containerTag,
        threshold: CONFIG.similarityThreshold,
        limit: CONFIG.maxMemories,
        searchMode: "hybrid" as const,
        ...(CONFIG.enableRerank ? { rerank: true } : {}),
      };
      const result = await withTimeout(
        (this.getClient().search.memories as (p: typeof searchParams) => Promise<{ results: Array<{ memory?: string; chunk?: string; similarity?: number; id?: string; metadata?: Record<string, unknown> }>; total: number; timing: number }>)(searchParams),
        TIMEOUT_MS
      );
      log("searchMemories: success", { count: result.results?.length || 0 });
      return { success: true as const, ...result };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      log("searchMemories: error", { error: errorMessage });
      return { success: false as const, error: errorMessage, results: [], total: 0, timing: 0 };
    }
  }

  async getProfile(containerTag: string, query?: string) {
    log("getProfile: start", { containerTag });
    try {
      const result = await withTimeout(
        this.getClient().profile({
          containerTag,
          q: query,
        }),
        TIMEOUT_MS
      );
      log("getProfile: success", { hasProfile: !!result?.profile });
      return { success: true as const, ...result };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      log("getProfile: error", { error: errorMessage });
      return { success: false as const, error: errorMessage, profile: null };
    }
  }

  async addMemory(
    content: string,
    containerTag: string,
    metadata?: { type?: MemoryType; tool?: string; [key: string]: unknown },
    entityContext?: string
  ) {
    log("addMemory: start", { containerTag, contentLength: content.length });
    try {
      const { createHash } = await import("node:crypto");
      const customId = createHash("sha256").update(content).digest("hex").slice(0, 16);
      const result = await withTimeout(
        this.getClient().add({
          content,
          containerTag,
          customId,
          ...(entityContext ? { entityContext } : {}),
          metadata: metadata as Record<string, string | number | boolean | string[]>,
        }),
        TIMEOUT_MS
      );
      log("addMemory: success", { id: result.id });
      return { success: true as const, ...result };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      log("addMemory: error", { error: errorMessage });
      return { success: false as const, error: errorMessage };
    }
  }

  async deleteMemory(memoryId: string, containerTag: string) {
    log("deleteMemory: start", { memoryId });
    try {
      await withTimeout(
        this.getClient().memories.forget({
          containerTag,
          id: memoryId,
          reason: "User requested deletion",
        }),
        TIMEOUT_MS
      );
      log("deleteMemory: success", { memoryId });
      return { success: true };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      log("deleteMemory: error", { memoryId, error: errorMessage });
      return { success: false, error: errorMessage };
    }
  }

  async listMemories(containerTag: string, limit = 20) {
    log("listMemories: start", { containerTag, limit });
    try {
      const result = await withTimeout(
        this.getClient().memories.list({
          containerTags: [containerTag],
          limit,
          order: "desc",
          sort: "createdAt",
          includeContent: true,
        }),
        TIMEOUT_MS
      );
      const resultTyped = result as { memories?: unknown[]; pagination?: unknown };
      log("listMemories: success", { count: resultTyped.memories?.length || 0 });
      return { success: true as const, ...resultTyped };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      log("listMemories: error", { error: errorMessage });
      return { success: false as const, error: errorMessage, memories: [], pagination: { currentPage: 1, totalItems: 0, totalPages: 0 } };
    }
  }

  async ingestConversation(
    conversationId: string,
    messages: ConversationMessage[],
    containerTags: string[],
    metadata?: Record<string, string | number | boolean>
  ) {
    log("ingestConversation: start", {
      conversationId,
      messageCount: messages.length,
      containerTags,
    });

    if (messages.length === 0) {
      return { success: false as const, error: "No messages to ingest" };
    }

    const uniqueTags = [...new Set(containerTags)].filter((tag) => tag.length > 0);
    if (uniqueTags.length === 0) {
      return { success: false as const, error: "At least one containerTag is required" };
    }

    // Try native v4/conversations endpoint first
    try {
      const nativeMessages = messages.map((m) => ({
        role: typeof m.role === "string" ? m.role : String(m.role),
        content:
          typeof m.content === "string"
            ? m.content
            : m.content
                .map((part) =>
                  part.type === "text" ? part.text : `[image] ${part.imageUrl.url}`
                )
                .join("\n"),
      }));

      const apiKey = SUPERMEMORY_API_KEY;
      const v4Response = await withTimeout(
        fetch("https://api.supermemory.ai/v4/conversations", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${apiKey}`,
          },
          body: JSON.stringify({
            conversationId,
            messages: nativeMessages,
            containerTags: uniqueTags,
            metadata: metadata ?? {},
          }),
        }),
        TIMEOUT_MS
      );

      if (v4Response.ok) {
        const v4Data = await v4Response.json() as { id?: string; status?: string };
        const id = v4Data.id ?? conversationId;
        log("ingestConversation: v4 success", { conversationId, status: v4Data.status });
        return {
          success: true as const,
          id,
          conversationId,
          status: (v4Data.status ?? "stored") as "stored" | "partial",
          storedMemoryIds: [id],
        };
      }
      // v4 endpoint not available — fall through to text-based approach
      log("ingestConversation: v4 not available, falling back", { status: v4Response.status });
    } catch (_v4Error) {
      log("ingestConversation: v4 request failed, falling back");
    }

    // Fallback: structured text format via addMemory
    const structuredContent = messages
      .map((m, idx) => {
        const role = typeof m.role === "string" ? m.role : String(m.role);
        const content =
          typeof m.content === "string"
            ? m.content
            : m.content
                .map((part) =>
                  part.type === "text" ? part.text : `[image] ${part.imageUrl.url}`
                )
                .join("\n");
        return `### Message ${idx + 1}\n**Role**: ${role}\n\n${content.trim()}`;
      })
      .join("\n\n---\n\n");

    const rawContent = `[Conversation ${conversationId}]\n\n${structuredContent}`;
    const content =
      rawContent.length > MAX_CONVERSATION_CHARS
        ? `${rawContent.slice(0, MAX_CONVERSATION_CHARS)}\n...[truncated]`
        : rawContent;

    const ingestMetadata = {
      type: "conversation" as const,
      conversationId,
      messageCount: messages.length,
      originalContainerTags: uniqueTags,
      ...metadata,
    };

    const savedIds: string[] = [];
    let firstError: string | null = null;

    for (const tag of uniqueTags) {
      const result = await this.addMemory(content, tag, ingestMetadata);
      if (result.success) {
        savedIds.push(result.id);
      } else if (!firstError) {
        firstError = result.error || "Failed to store conversation";
      }
    }

    if (savedIds.length === 0) {
      log("ingestConversation: error", { conversationId, error: firstError });
      return {
        success: false as const,
        error: firstError || "Failed to ingest conversation",
      };
    }

    const status =
      savedIds.length === uniqueTags.length ? "stored" : "partial";
    const response: ConversationIngestResponse = {
      id: savedIds[0]!,
      conversationId,
      status,
    };

    log("ingestConversation: success", {
      conversationId,
      status,
      storedCount: savedIds.length,
      requestedCount: uniqueTags.length,
    });

    return {
      success: true as const,
      ...response,
      storedMemoryIds: savedIds,
    };
  }

}

export const supermemoryClient = new SupermemoryClient();
