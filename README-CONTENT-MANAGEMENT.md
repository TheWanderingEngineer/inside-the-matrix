# Inside The Matrix - Content Management Guide

## 📝 How to Add a New Article

### Step 1: Open the posts data file
Navigate to `src/data/posts.ts`

### Step 2: Add a new post object
Add a new entry to the `posts` array following this structure:

```typescript
{
  id: "4", // Increment the ID number
  title: "Your Article Title Here",
  summary: "A brief 1-2 sentence summary that appears on the card",
  content: `
# Your Article Title

Your full article content goes here in Markdown format.

## Subheadings

You can use all standard Markdown features:
- Bullet lists
- **Bold text**
- *Italic text*
- \`inline code\`

### Code Blocks

\`\`\`python
def hello_world():
    print("Hello, World!")
\`\`\`

### Images

To add images within your article:

![Alt text](/images/your-image.jpg)

## More Sections

Continue writing your content...
  `,
  tags: ["Tag1", "Tag2", "Tag3"], // Add relevant tags
  image: "/images/your-thumbnail.jpg", // Path to thumbnail
  date: "2024-01-30", // Publication date (YYYY-MM-DD)
  readTime: "7 min read" // Estimated reading time
}
```

### Step 3: Save the file
The new article will appear automatically on the homepage!

---

## 🖼️ How to Add or Change Images

### For Article Thumbnails (Card Images)

1. **Add your image to the images folder:**
   - Place your image file in `/public/images/`
   - Supported formats: JPG, PNG, WebP, SVG
   - Recommended size: 1920x1080px (16:9 aspect ratio)

2. **Reference the image in your post:**
   ```typescript
   image: "/images/your-image-name.jpg"
   ```

### For Inline Images Within Articles

1. **Add your image to the images folder:**
   - Place your image in `/public/images/`

2. **Reference it in your Markdown content:**
   ```markdown
   ![Description of image](/images/your-inline-image.jpg)
   ```

### Image Best Practices

- **Thumbnail images**: 1920x1080px for optimal display
- **Inline images**: Keep under 2MB for fast loading
- **File naming**: Use lowercase with hyphens (e.g., `neural-network-diagram.jpg`)
- **Alt text**: Always provide descriptive alt text for accessibility

---

## 🏷️ How to Manage Tags

Tags are automatically extracted from all posts and displayed as filters.

### Adding Tags to a Post

In your post object, add tags as an array of strings:

```typescript
tags: ["Deep Learning", "Python", "Tutorial"]
```

### Tag Guidelines

- Use **2-4 tags** per post for best results
- Be consistent with capitalization (e.g., always "Machine Learning", not "machine learning")
- Create new tags sparingly - reuse existing tags when possible
- Popular tags:
  - "Deep Learning"
  - "Neural Networks"
  - "NLP"
  - "MLOps"
  - "Python"
  - "Tutorial"
  - "Research"

---

## 📐 Content Formatting Tips

### Markdown Features Supported

- **Headers**: Use `#` for H1, `##` for H2, etc.
- **Bold**: `**bold text**`
- **Italic**: `*italic text*`
- **Code inline**: `` `code` ``
- **Code blocks**: Use triple backticks with language
- **Lists**: Use `-` or `1.` for bullet/numbered lists
- **Links**: `[text](url)`
- **Images**: `![alt](path)`
- **Quotes**: Use `>` for blockquotes

### Code Block Syntax

For syntax highlighting, specify the language:

\`\`\`python
# Python code here
def example():
    pass
\`\`\`

\`\`\`javascript
// JavaScript code here
const example = () => {};
\`\`\`

Supported languages: Python, JavaScript, TypeScript, Bash, JSON, HTML, CSS, and more.

---

## 🎨 Design Recommendations

### Writing Style
- Keep summaries concise (under 160 characters)
- Use clear, descriptive titles
- Break content into sections with headers
- Include code examples where relevant

### Visual Hierarchy
- Start with an engaging introduction
- Use H2 (`##`) for main sections
- Use H3 (`###`) for subsections
- Keep paragraphs short (3-5 sentences)

### Image Guidelines
- Use high-quality, relevant images
- Ensure images have good contrast
- Optimize file sizes before uploading
- Match the Matrix-inspired aesthetic (green/cyan tones work well)

---

## 🚀 Quick Reference

### File Locations
- **Posts data**: `src/data/posts.ts`
- **Images**: `/public/images/`
- **Styles**: `src/index.css`

### Example Post Structure
```typescript
{
  id: "unique-id",
  title: "Engaging Title",
  summary: "Brief, compelling summary",
  content: `# Full Markdown Content`,
  tags: ["Tag1", "Tag2"],
  image: "/images/thumbnail.jpg",
  date: "YYYY-MM-DD",
  readTime: "X min read"
}
```

---

## 💡 Tips for Great Content

1. **Start Strong**: Hook readers with an interesting opening
2. **Use Visuals**: Break up text with images and code examples
3. **Be Consistent**: Maintain a regular posting schedule
4. **Tag Wisely**: Use relevant, searchable tags
5. **Proofread**: Check for typos and formatting issues
6. **Optimize Images**: Compress images to keep site fast

---

## ❓ Troubleshooting

### Image not showing?
- Check the file path starts with `/images/`
- Ensure the file exists in `/public/images/`
- Verify the filename matches exactly (case-sensitive)

### Code block not highlighting?
- Make sure you specify the language after the triple backticks
- Check that the language name is spelled correctly

### Post not appearing?
- Ensure the post object is properly formatted
- Check for missing commas in the array
- Verify the `id` is unique

---

Happy blogging! 🚀
