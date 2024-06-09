import PyPDF2
from typing import List

def pdf_to_text(pdf_file: str) -> List[str]:
    text = ""
    with open(pdf_file, 'rb') as file:
        reader = PyPDF2.PdfReader(file, strict=False)
        pdf_text = []

        for page in reader.pages:
            content = page.extract_text()
            pdf_text.append(content)
    return pdf_text


# text_content = pdf_to_text()
# print(text_content)
# text_content = pdf_to_text('../assets/sample')
# for text in text_content:
#     print(text)
# import PyPDF2
# from collections import Counter
#
#
# def pdf_to_text(pdf_file: str) -> [str]:
#     text = ""
#     with open(pdf_file, 'rb') as file:
#         reader = PyPDF2.PdfReader(file, strict=False)
#         pdf_text = []
#
#         for page in reader.pages:
#             content = page.extract_text()
#             pdf_text.append(content)
#     return pdf_text
#
#
# def get_word_frequency(text_content: [str]) -> Counter:
#     words = []
#     for text in text_content:
#
#         words.extend(text.split())
#
#
#     word_frequency = Counter(words)
#     return word_frequency
#
#
# def print_sorted_word_frequency(word_frequency: Counter):
#
#     sorted_word_frequency = dict(sorted(word_frequency.items(), key=lambda x: x[1], reverse=True))
#
#     for word, freq in sorted_word_frequency.items():
#         print(f"{word}: {freq}")
#
#
# text_content = pdf_to_text('sample')
# for text in text_content:
#     print(text)
# word_frequency = get_word_frequency(text_content)
# print_sorted_word_frequency(word_frequency)
