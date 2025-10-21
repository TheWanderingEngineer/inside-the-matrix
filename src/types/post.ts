export type Post = {
  title: string;
  summary: string;
  date?: string;      // ISO yyyy-mm-dd
  tags?: string[];
  content: string;    // markdown
};