from pickle import load
from keras.models import load_model
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

def generate_seq(model, vocab_to_int,int_to_vocab, seq_length, seed_text, n_chars):
	in_text = seed_text
	for _ in range(n_chars):
		encoded = [vocab_to_int[char] for char in in_text]
		encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
		encoded = to_categorical(encoded, num_classes=len(vocab_to_int))
		yhat = model.predict_classes(encoded, verbose=0)
		out_char = int_to_vocab[yhat[0]]
		in_text += out_char
	return in_text

model = load_model('LSTM/model00000010.h5')
vocab_to_int = load(open('LSTM/vocab_to_int.pkl', 'rb'))
int_to_vocab = load(open('LSTM/int_to_vocab.pkl','rb'))

text = generate_seq(model,vocab_to_int,int_to_vocab,10,'JULIET',150)
print(text)
