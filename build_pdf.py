import os
from markdown_pdf import MarkdownPdf, Section

def generate_pdf():
    # Read the markdown file
    with open('research_paper.md', 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    print("Initializing PDF generator...")
    # Initialize PDF builder
    pdf = MarkdownPdf(toc_level=2)
    
    # We can add the content as a single section
    pdf.add_section(Section(md_content, toc=False))
    
    # Save the file
    out_path = 'research_paper.pdf'
    pdf.meta["title"] = "Brain Region Activation Does Not Predict Social Media Virality"
    pdf.meta["author"] = "Fayas P Sulfikkar"
    pdf.save(out_path)
    print(f"Generated {out_path} successfully!")

if __name__ == "__main__":
    generate_pdf()
