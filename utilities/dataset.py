import os
import pandas as pd
import numpy as np

def load_dataset(base_path):
    """Load the WikiPhrase dataset.
    
    Parameters
    ----------
    base_path : str
        The path to the folder which contains the WikiPhrase dataset.
    
    Returns
    -------
    DataFrame
        A DataFrame in which each row describes annotated phrases.
    """
    data_path = os.path.join('..', '..', 'data', 'wikiphrase')
    df_annotations = pd.read_csv(os.path.join(base_path, 'annotations.csv'))
    df_docs = pd.read_json(os.path.join(base_path, 'wikinews-docs.json'))
    df_entities = pd.read_json(os.path.join(base_path, 'wikinews-entities.json'))
    df_wikipedia = pd.read_json(os.path.join(base_path, 'wikipedia-entities.json'))
    df_annotations = pd.merge(df_annotations, df_docs, on='doc_id', how='inner')
    df_annotations = pd.merge(df_annotations, df_wikipedia, on='entity', how='outer')
    df_annotations = pd.merge(df_annotations, df_entities, on=['doc_id', 'entity'], how='inner')
    df_annotations = df_annotations.set_index(['doc_id', 'entity'])
    return df_annotations

def generate_embeddings(tokens, embedding_size, random_seed=0):
    np.random.seed(random_seed)
    num_tokens = len(tokens)
    weights = np.random.normal(0, 0.08, (num_tokens, embedding_size))
    df_embeddings = pd.DataFrame(weights)
    df_embeddings.index = tokens
    return df_embeddings

class Embeddings:
    
    def __init__(self, df_current_embeddings, add_unk_token=True, add_pad_token=True, add_start_token=True, add_end_token=True, random_seed=0):
        # Fix the seed such that the same additional embeddings are generated every time
        np.random.seed(random_seed)
        df_markers = []
        embedding_size = df_current_embeddings.shape[1]
        if add_pad_token:
            df_markers.append([0.] * embedding_size + ['__PAD__'])
        if add_unk_token:
            df_markers.append([np.random.normal(0., 0.08)] * embedding_size + ['__UNK__'])
        if add_start_token:
            df_markers.append([np.random.normal(0., 0.08)] * embedding_size + ['__START__'])
        if add_end_token:
            df_markers.append([np.random.normal(0., 0.08)] * embedding_size + ['__END__'])
        df_embeddings = df_current_embeddings.copy()
        if len(df_markers) > 0:
            df_markers = pd.DataFrame(df_markers)
            df_markers = df_markers.set_index(df_markers.columns[-1])
            df_embeddings = pd.concat([df_markers, df_embeddings], axis=0)
        self.df_embeddings = df_embeddings
        self.vocab = {word: index for index, word in enumerate(df_embeddings.index)}
        self.vocab_inv = {index: word for word, index in self.vocab.items()}
    
    def get_weights(self):
        return self.df_embeddings.values
    
    def get_vocab(self):
        return self.vocab
    
    def get_inverse_vocab(self):
        return self.vocab_inv
    
    def lookup(self, token):
        if token in self.vocab:
            return self.vocab[token]
        elif '__UNK__' in self.vocab:
            return self.vocab['__UNK__']
        else:
            raise NotImplementedError('Token "{}" is not found and no "unknown token" marker is available in the vocabulary.'.format(token))
    
    def inverse_lookup(self, identifier):
        if identifier in self.vocab_inv:
            return self.vocab_inv[identifier]
        else:
            raise ValueError('Identifier "{}" is not available in the inverse vocabulary.'.format(identifier))

class LowercaseTransformer:
    
    def __init__(self):
        pass
    
    def __call__(self, text):
        return text.lower()
            
class Tokenizer:
    
    def __init__(self, embeddings, subtokenizer, transformers=[], add_start_token=True, add_end_token=True):
        self.embeddings = embeddings
        self.tokenizer = subtokenizer
        self.transformers = transformers
        self.add_start_token = add_start_token
        self.add_end_token = add_end_token
        
    def normalize(self, token):
        normalized_token = token
        for transformer in self.transformers:
            normalized_token = transformer(normalized_token)
        return normalized_token
    
    def __call__(self, text):
        tokens = self.tokenizer(text)
        normalized_tokens = [self.normalize(token) for token in tokens]
        if self.add_start_token:
            tokens = ['__START__'] + tokens
            normalized_tokens = ['__START__'] + normalized_tokens
        if self.add_end_token:
            tokens = tokens + ['__END__']
            normalized_tokens = normalized_tokens + ['__END__']
        ids = [self.embeddings.lookup(token) for token in normalized_tokens]
        return {
            'tokens': tokens,
            'normalized_tokens': normalized_tokens,
            'ids': ids
        }

class TextEncoder:
    
    def __init__(self, word_tokenizer=None, char_tokenizer=None):
        self.word_tokenizer = word_tokenizer
        self.char_tokenizer = char_tokenizer
        
    def remove_markers(self, text):
        if text.startswith('__') and text.endswith('__'):
            return ''
        else:
            return text

    def __call__(self, **texts):
        output = {}
        for text_label, text_value in texts.items():
            if self.word_tokenizer is not None:
                word_tokenizer_output = self.word_tokenizer(text_value)
                output['{}__word_tokens'.format(text_label)] = word_tokenizer_output['normalized_tokens']
                output['{}__word_ids'.format(text_label)] = np.array(word_tokenizer_output['ids']).astype('i')
                if self.char_tokenizer is not None:
                    char_tokenizer_output = [self.char_tokenizer(self.remove_markers(token)) for token in word_tokenizer_output['tokens']]
                    output['{}__char_tokens'.format(text_label)] = [output['normalized_tokens'] for output in char_tokenizer_output]
                    output['{}__char_ids'.format(text_label)] = [np.array(output['ids']).astype('i') for output in char_tokenizer_output]
        return output