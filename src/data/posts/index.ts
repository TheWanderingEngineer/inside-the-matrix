import type { Post } from "@/types/post";

// Eagerly import every .ts file in this folder (except index.ts)
const modules = import.meta.glob("./*.ts", { eager: true });

type PostWithId = Post & { id: string };

const postsMap: Record<string, Post> = Object.fromEntries(
  Object.entries(modules).map(([path, mod]) => {
    const id = path.replace("./", "").replace(".ts", "");   // filename = id
    // @ts-expect-error default export exists
    return [id, (mod as any).default as Post];
  })
);

export function getAllPosts(): PostWithId[] {
  return Object.entries(postsMap)
    .map(([id, p]) => ({ id, ...p }))
    .sort((a, b) => (b.date || "").localeCompare(a.date || "")); // newest first
}

export function getPostById(id: string): PostWithId | undefined {
  const p = postsMap[id];
  return p ? ({ id, ...p }) : undefined;
}