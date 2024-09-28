from transformers import MarianMTModel, MarianTokenizer 

model_name_en_to_ru = "Helsinki-NLP/opus-mt-en-ru"
model_en_to_ru = MarianMTModel.from_pretrained(model_name_en_to_ru)
tokenizer_en_to_ru = MarianTokenizer.from_pretrained(model_name_en_to_ru)
model_name_ru_to_en = 'Helsinki-NLP/opus-mt-ru-en'
model_ru_to_en = MarianMTModel.from_pretrained(model_name_ru_to_en)
tokenizer_ru_to_en = MarianTokenizer.from_pretrained(model_name_ru_to_en)

def translate_en_to_ru(text):
    tokenizer_text = tokenizer_en_to_ru([text], return_tensors='pt', padding=True)
    translated_text = model_en_to_ru.generate(**tokenizer_text)
    return translated_text 

def translate_ru_to_en(text):
    tokenizer_text = tokenizer_ru_to_en([text], return_tensors='pt', padding=True)
    translated_text = model_ru_to_en.generate(**tokenizer_text)
    return translated_text 

english_text = "Hello world!"
translated_to_russian = translate_en_to_ru(english_text)
print(f'Перевод на русский: {translated_to_russian}')

russian_text = 'Привет мир!'
translated_to_english = translate_ru_to_en(russian_text)
print(f'Перевод на английский: {translated_to_english}')