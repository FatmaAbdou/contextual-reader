from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

def create_test_pdf(filename="test.pdf"):
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter

    # Write a title and some sentences
    c.drawString(100, height - 100, "This is a test PDF for the Contextual Reader app.")
    c.drawString(100, height - 130, "The app uses RAG to answer questions about this document.")
    c.drawString(100, height - 160, "Question: What does RAG stand for?")
    c.drawString(100, height - 190, "Answer: Retrieval-Augmented Generation.")

    c.save()
    print(f"Test PDF '{filename}' created successfully!")

if __name__ == "__main__":
    create_test_pdf()