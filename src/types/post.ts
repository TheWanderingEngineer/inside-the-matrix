export type Post = {
  title: string;
  summary: string;
  content: string;
  tags: string[];
  image: string;
  date: string;      // ISO
  readTime: string;  // e.g., "8 min read"
};