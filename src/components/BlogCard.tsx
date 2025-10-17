import { Calendar, Clock } from "lucide-react";
import { Card } from "@/components/ui/card";
import { BlogPost } from "@/data/posts";
import { Link } from "react-router-dom";

interface BlogCardProps {
  post: BlogPost;
  onTagClick: (tag: string) => void;
}

const BlogCard = ({ post, onTagClick }: BlogCardProps) => {
  return (
    <Card className="overflow-hidden blog-card border-border hover:border-primary/50 transition-all duration-300">
      <Link to={`/post/${post.id}`}>
        <div className="aspect-video overflow-hidden bg-muted">
        <img
          src={`${import.meta.env.BASE_URL}${post.image}`}
          alt={post.title}
          className="w-full h-full object-cover transition-transform duration-500 hover:scale-110"
        />
        </div>
      </Link>
      
      <div className="p-6 space-y-4">
        <div className="flex items-center gap-4 text-sm text-muted-foreground">
          <div className="flex items-center gap-1">
            <Calendar className="h-4 w-4" />
            <span>{new Date(post.date).toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' })}</span>
          </div>
          <div className="flex items-center gap-1">
            <Clock className="h-4 w-4" />
            <span>{post.readTime}</span>
          </div>
        </div>

        <Link to={`/post/${post.id}`}>
          <h2 className="text-2xl font-bold link-underline inline-block">
            {post.title}
          </h2>
        </Link>

        <p className="text-muted-foreground line-clamp-2">
          {post.summary}
        </p>

        <div className="flex flex-wrap gap-2 pt-2">
          {post.tags.map((tag) => (
            <button
              key={tag}
              onClick={() => onTagClick(tag)}
              className="tag"
            >
              {tag}
            </button>
          ))}
        </div>
      </div>
    </Card>
  );
};

export default BlogCard;
