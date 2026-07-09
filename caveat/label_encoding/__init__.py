from caveat.label_encoding.base import BaseLabelEncoder, tokenize
from caveat.label_encoding.onehot import OneHotAttributeEncoder
from caveat.label_encoding.table_encoder import TableEncoder
from caveat.label_encoding.tokenise import TokenAttributeEncoder

library = {
    "onehot": OneHotAttributeEncoder,
    "tokens": TokenAttributeEncoder,
    "table": TableEncoder,
}
