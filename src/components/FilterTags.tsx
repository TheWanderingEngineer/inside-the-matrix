import { X } from "lucide-react";
import { Button } from "@/components/ui/button";

interface FilterTagsProps {
  allTags: string[];
  selectedTags: string[];
  toggleTag: (tag: string) => void;
}

const FilterTags = ({ allTags, selectedTags, toggleTag }: FilterTagsProps) => {
  if (allTags.length === 0) return null;

  return (
    <div className="flex flex-wrap items-center gap-2 max-w-5xl mx-auto">
      <span className="text-sm font-medium text-muted-foreground">Filter by tag:</span>
      {allTags.map((tag) => {
        const isSelected = selectedTags.includes(tag);
        return (
          <Button
            key={tag}
            variant={isSelected ? "default" : "outline"}
            size="sm"
            onClick={() => toggleTag(tag)}
            className={`transition-all duration-200 ${
              isSelected ? "matrix-glow" : ""
            }`}
          >
            {tag}
            {isSelected && <X className="ml-1 h-3 w-3" />}
          </Button>
        );
      })}
    </div>
  );
};

export default FilterTags;
