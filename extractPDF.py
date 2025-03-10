from PyPDF2 import PdfReader, PdfWriter

# Function to extract pages from a PDF
def extract_pages(input_pdf_path, output_pdf_path, page_numbers):
    reader = PdfReader(input_pdf_path)
    writer = PdfWriter()

    for page_number in page_numbers:
        writer.add_page(reader.pages[page_number - 1])

    with open(output_pdf_path, 'wb') as output_pdf:
        writer.write(output_pdf)

# Example usage
input_pdf_path = r"F:\BaiduNetdiskDownload\16-月落洼（5人）\剧本\吾特+睿司.pdf"
output_pdf_path = r"F:\BaiduNetdiskDownload\16-月落洼（5人）\剧本\吾特.pdf"
page_numbers = [1, 3, 5]  # List of page numbers to extract (1-based index)

extract_pages(input_pdf_path, output_pdf_path, range(1, 11)) 