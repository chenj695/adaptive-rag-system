import json
import re
from pathlib import Path
from typing import Dict, List


class ReportsProcessor:
    def __init__(self, use_serialized_tables: bool = False, 
                 serialized_tables_instead_of_markdown: bool = False):
        self.report_data = None
        self.use_serialized_tables = use_serialized_tables
        self.serialized_tables_instead_of_markdown = serialized_tables_instead_of_markdown
        
    def process_report(self, report_data: Dict) -> Dict:
        """Process a single report and return simplified format."""
        self.report_data = report_data
        processed_pages = []
        
        for page in report_data.get("content", []):
            page_text = self.process_page(page["page"])
            processed_pages.append({
                "page": page["page"],
                "text": page_text
            })
        
        return {
            "metainfo": report_data.get("metainfo", {}),
            "pages": processed_pages
        }
    
    def process_page(self, page_number: int) -> str:
        """Process a single page and return markdown text."""
        page_data = self._get_page_data(page_number)
        if not page_data or "content" not in page_data:
            return ""
        
        blocks = page_data["content"]
        filtered_blocks = self._filter_blocks(blocks)
        final_blocks = self._apply_formatting_rules(filtered_blocks)
        
        if final_blocks:
            final_blocks[0] = final_blocks[0].lstrip()
            final_blocks[-1] = final_blocks[-1].rstrip()
        
        return "\n".join(final_blocks)
    
    def _get_page_data(self, page_number):
        """Get page data by page number."""
        all_pages = self.report_data.get("content", [])
        for page in all_pages:
            if page.get("page") == page_number:
                return page
        return None
    
    def _filter_blocks(self, blocks):
        """Filter out ignored block types."""
        ignored_types = {"page_footer", "picture"}
        return [b for b in blocks if b.get("type") not in ignored_types]
    
    def _clean_text(self, text):
        """Clean text from docling artifacts."""
        command_mapping = {
            'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
            'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
            'period': '.', 'comma': ',', 'colon': ":", 'hyphen': "-",
            'percent': '%', 'dollar': '$', 'space': ' ', 'plus': '+',
            'minus': '-', 'slash': '/', 'asterisk': '*',
            'lparen': '(', 'rparen': ')', 'parenright': ')',
            'parenleft': '(', 'wedge.1_E': '',
        }
        recognized_commands = "|".join(command_mapping.keys())
        slash_command_pattern = rf"/({recognized_commands})(\.pl\.tnum|\.tnum\.pl|\.pl|\.tnum|\.case|\.sups)"
        
        def replace_command(match):
            base_command = match.group(1)
            replacement = command_mapping.get(base_command)
            return replacement if replacement is not None else match.group(0)
        
        text = re.sub(slash_command_pattern, replace_command, text)
        text = re.sub(r'glyph<[^>]*>', '', text)
        text = re.sub(r'/([A-Z])\.cap', r'\1', text)
        return text
    
    def _apply_formatting_rules(self, blocks):
        """Apply formatting rules to blocks."""
        final_blocks = []
        
        for i, block in enumerate(blocks):
            block_type = block.get("type")
            text = block.get("text", "")
            text = self._clean_text(text)
            
            if not text.strip():
                continue
            
            if block_type == "section_header":
                final_blocks.append(f"## {text}\n")
            elif block_type in ("paragraph", "text"):
                final_blocks.append(f"{text}\n")
            elif block_type == "table":
                table_md = self._get_table_by_id(block.get("table_id"))
                if table_md:
                    final_blocks.append(f"{table_md}\n")
            elif block_type == "list_item":
                final_blocks.append(f"- {text}\n")
            elif block_type == "footnote":
                final_blocks.append(f"[^{i}] {text}\n")
            elif block_type == "page_header":
                final_blocks.append(f"*{text}*\n")
            else:
                final_blocks.append(f"{text}\n")
        
        return final_blocks
    
    def _get_table_by_id(self, table_id):
        """Get table markdown by ID."""
        for t in self.report_data.get("tables", []):
            if t.get("table_id") == table_id:
                if self.use_serialized_tables:
                    return self._get_serialized_table_text(t, self.serialized_tables_instead_of_markdown)
                return t.get("markdown", "")
        return ""
    
    def _get_serialized_table_text(self, table, serialized_tables_instead_of_markdown):
        """Get serialized table text."""
        if not table.get("serialized"):
            return table.get("markdown", "")
        
        info_blocks = table["serialized"].get("information_blocks", [])
        text_blocks = [block["information_block"] for block in info_blocks]
        serialized_text = "\n".join(text_blocks)
        
        if serialized_tables_instead_of_markdown:
            return serialized_text
        else:
            markdown = table.get("markdown", "")
            return f"{markdown}\nDescription of the table entities:\n{serialized_text}"
    
    def process_reports(self, reports_dir: Path, output_dir: Path):
        """Process all reports in directory."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for report_path in reports_dir.glob("*.json"):
            with open(report_path, 'r', encoding='utf-8') as f:
                report_data = json.load(f)
            
            processed_report = self.process_report(report_data)
            output_path = output_dir / report_path.name
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(processed_report, f, indent=2, ensure_ascii=False)
        
        print(f"Processed reports saved to {output_dir}")
    
    def export_reports_to_markdown(self, reports_dir: Path, output_dir: Path):
        """Export reports to markdown files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for report_path in reports_dir.glob("*.json"):
            with open(report_path, 'r', encoding='utf-8') as f:
                report_data = json.load(f)
            
            processed_report = self.process_report(report_data)
            document_text = ""
            
            for page in processed_report['pages']:
                document_text += f"\n\n---\n\n# Page {page['page']}\n\n"
                document_text += page['text']
            
            report_name = report_data['metainfo']['sha1_name']
            with open(output_dir / f"{report_name}.md", "w", encoding="utf-8") as f:
                f.write(document_text)
        
        print(f"Markdown exports saved to {output_dir}")
