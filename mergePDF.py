from PyPDF2 import PdfMerger

def merge_pdfs(pdf_list, output_filename="merged_output.pdf"):
    merger = PdfMerger()
    
    for pdf in pdf_list:
        merger.append(pdf)
    
    merger.write(output_filename)
    merger.close()
    print(f"Merged PDF saved as {output_filename}")

# Example usage
pdf_files = [r"C:\Users\wangs\Downloads\58-201-Renter's insurance id.pdf", r"C:\Users\wangs\Downloads\58-201-Renter's insurance.pdf"]  # Replace with your file names
merge_pdfs(pdf_files, r"C:\Users\wangs\Downloads\58-201-Renter's insurance combined.pdf")
