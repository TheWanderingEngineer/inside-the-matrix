import { useState, useMemo } from "react";
import SearchBar from "@/components/SearchBar";
import FilterTags from "@/components/FilterTags";
import BlogCard from "@/components/BlogCard";
import { posts } from "@/data/posts";

const Index = () => {
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedTags, setSelectedTags] = useState<string[]>([]);

  // Get all unique tags
  const allTags = useMemo(() => {
    const tags = new Set<string>();
    posts.forEach((post) => post.tags.forEach((tag) => tags.add(tag)));
    return Array.from(tags).sort();
  }, []);

  // Filter posts based on search and tags
  const filteredPosts = useMemo(() => {
    return posts.filter((post) => {
      // Search filter
      const matchesSearch =
        searchQuery === "" ||
        post.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
        post.summary.toLowerCase().includes(searchQuery.toLowerCase()) ||
        post.tags.some((tag) =>
          tag.toLowerCase().includes(searchQuery.toLowerCase())
        ) ||
        post.content.toLowerCase().includes(searchQuery.toLowerCase());

      // Tag filter
      const matchesTags =
        selectedTags.length === 0 ||
        selectedTags.every((tag) => post.tags.includes(tag));

      return matchesSearch && matchesTags;
    });
  }, [searchQuery, selectedTags]);

  const toggleTag = (tag: string) => {
    setSelectedTags((prev) =>
      prev.includes(tag) ? prev.filter((t) => t !== tag) : [...prev, tag]
    );
  };

  const handleTagClickFromCard = (tag: string) => {
    if (!selectedTags.includes(tag)) {
      setSelectedTags([...selectedTags, tag]);
    }
  };

  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="py-16 md:py-24 border-b">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8">
          <div className="max-w-3xl mx-auto text-center space-y-6">
            <h1 className="text-5xl md:text-6xl lg:text-7xl font-bold leading-tight">
              <span className="bg-gradient-to-r from-primary via-accent to-primary bg-clip-text text-transparent matrix-glow">
                AI & Machine Learning
              </span>
              <br />
              <span className="text-foreground">Insights</span>
            </h1>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              Deep dives into artificial intelligence, machine learning, and the future of technology
            </p>
          </div>
        </div>
      </section>

      {/* Search and Filter Section */}
      <section className="py-8 border-b bg-muted/30">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8 space-y-6">
          <SearchBar searchQuery={searchQuery} setSearchQuery={setSearchQuery} />
          <FilterTags
            allTags={allTags}
            selectedTags={selectedTags}
            toggleTag={toggleTag}
          />
        </div>
      </section>

      {/* Blog Posts Grid */}
      <section className="py-12 md:py-16">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8">
          {filteredPosts.length === 0 ? (
            <div className="text-center py-16">
              <p className="text-xl text-muted-foreground">
                No posts found matching your criteria
              </p>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 max-w-7xl mx-auto">
              {filteredPosts.map((post) => (
                <BlogCard
                  key={post.id}
                  post={post}
                  onTagClick={handleTagClickFromCard}
                />
              ))}
            </div>
          )}
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t py-8 mt-16">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center text-sm text-muted-foreground">
            <p>© 2024 Inside The Matrix. All rights reserved.</p>
            <p className="mt-2">Built with React, TypeScript, and Tailwind CSS</p>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Index;
