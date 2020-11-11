import sentencepiece as spm


class CharTokenizer(object):
    @classmethod
    def init(cls, train_input_file='data/crime-and-punishment.txt',
             pad_token_id=0, bos_token_id=1, eos_token_id=2, unk_token_id=3,
             prefix="sentencepiece_models/cp", vocab_size=300, character_coverage=0.9995,
             user_defined_symbols="<n>", model_type='char', normalization_rule_name="nfkc_cf"):

        prefix += f"-{model_type}"
        template = ("--input={} --pad_id={} --bos_id={} --eos_id={} --unk_id={} "
                    "--model_prefix={} --vocab_size={} --character_coverage={} "
                    "--user_defined_symbols={} --model_type={} --normalization_rule_name={}")
        cmd = template.format(
            train_input_file, pad_token_id, bos_token_id, eos_token_id, unk_token_id,
            prefix, vocab_size, character_coverage,
            user_defined_symbols, model_type, normalization_rule_name
        )

        spm.SentencePieceTrainer.Train(cmd)

    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)
        self.vocab_size = self.sp.vocab_size()
        self.pad_token_id = self.sp.pad_id()
        self.unk_token_id = self.sp.unk_id()

    def encode(self, text):
        return self.sp.EncodeAsIds(text)

    def decode(self, sequence):
        return self.sp.DecodeIds(sequence)

    def tokenize(self, text):
        return self.sp.EncodeAsPieces(text)

    def __call__(self, text):
        return self.encode(text)


if __name__ == "__main__":
    sentencepiece_models_dir = "sentencepiece_models"
    CharTokenizer.init(model_type="char", vocab_size=300)