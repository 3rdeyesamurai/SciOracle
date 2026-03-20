import PyPDF2
reader = PyPDF2.PdfReader('c:\\Users\\yungk\\Downloads\\SciOracle.pdf')
text = ''.join([page.extract_text() for page in reader.pages])
with open('scioracle_extracted.txt', 'w', encoding='utf-8') as f:
    f.write(text)
