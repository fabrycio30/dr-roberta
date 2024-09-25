# -*- coding: utf-8 -*-
"""Legal-Roberta-pretraining.ipynb
Bumbabert*** foi pré-treinado com um modelo pequeno de 84 milhões de parâmetros, usando o mesmo número de camadas e cabeças que o DistilBert, ou seja, 6 camadas, tamanho oculto de 768 e 12 cabeças de atenção. ***Bumbabert*** é então ajustado para uma tarefa downstream de modelagem de linguagem mascarada.




# Fase 1: Carregando os dados
"""

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd

data = pd.read_csv('/content/drive/MyDrive/projetos/dados/pos-bumbabert/jf_p3d_pp.csv')

df = data.sample(1000)

"""#Fase 2: Instalando Hugging Face transformers

"""

!pip install Transformers
!pip install --upgrade accelerate
from accelerate import Accelerator

"""#Fase 3: Treinando o tokenizer do zero

"""

from pathlib import Path

from tokenizers import ByteLevelBPETokenizer

paths = [str(x) for x in Path(".").glob("**/*.txt")]

# Read the content from the files, ignoring or replacing invalid characters
file_contents = []
for path in paths:
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as file:
            file_contents.append(file.read())
    except Exception as e:
        print(f"Error reading {path}: {e}")

# Join the contents into a single string
text = "\n".join(file_contents)

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

# Customize training
tokenizer.train_from_iterator([text], vocab_size=52_000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

# Concatenar os textos da coluna 'text'
text = "\n".join(df['text'])

len(text)

from tokenizers import ByteLevelBPETokenizer

# Inicializar o tokenizer
tokenizer = ByteLevelBPETokenizer()

# Treinar o tokenizer
tokenizer.train_from_iterator([text], vocab_size=52_000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

"""#Fase 4: Salvando ao arquivos em disco

"""

import os
token_dir = '/content/drive/MyDrive/projetos/bumbabert/drRoBerta/model1'
if not os.path.exists(token_dir):
  os.makedirs(token_dir)
tokenizer.save_model('/content/drive/MyDrive/projetos/bumbabert/drRoBerta/model1')

"""#Step 5: Loading the trained tokenizer files"""

from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
path = "/content/drive/MyDrive/projetos/bumbabert/drRoBerta/model1"
tokenizer = ByteLevelBPETokenizer(
    f"{path}/vocab.json",
    f"{path}/merges.txt",
)

tokenizer.encode("Excelentissimo senhor juiz").tokens

tokenizer.encode("Excelentissimo senhor juiz")

tokenizer._tokenizer.post_processor = BertProcessing(
    ("</s>", tokenizer.token_to_id("</s>")),
    ("<s>", tokenizer.token_to_id("<s>")),
)
tokenizer.enable_truncation(max_length=512)

"""#Step 6: Checking Resource Constraints: GPU and NVIDIA"""

!nvidia-smi

#@title Checking that PyTorch Sees CUDA
import torch
torch.cuda.is_available()

"""#Step 7: Defining the configuration of the model"""

from transformers import RobertaConfig

config = RobertaConfig(
    vocab_size=30000,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
    #padding_idx=0
)

print(config)

"""#Step 8: Reloading the tokenizer in transformers"""

from transformers import RobertaTokenizer
tokenizer = RobertaTokenizer.from_pretrained(path, max_length=512)

"""#Step 9: Initializing a model from scratch"""

from transformers import RobertaForMaskedLM

model = RobertaForMaskedLM(config=config)
print(model)

"""## Exploring the parameters"""

print(model.num_parameters())

LP=list(model.parameters())
lp=len(LP)
print(lp)
for p in range(0,lp):
  print(LP[p])

#Shape of each tensor in the model
LP = list(model.parameters())
for i, tensor in enumerate(LP):
    print(f"Shape of tensor {i}: {tensor.shape}")

#counting the parameters
np=0
for p in range(0,lp):#number of tensors
  PL2=True
  try:
    L2=len(LP[p][0]) #check if 2D
  except:
    L2=1             #not 2D but 1D
    PL2=False
  L1=len(LP[p])
  L3=L1*L2
  np+=L3             # number of parameters per tensor
  if PL2==True:
    print(p,L1,L2,L3)  # displaying the sizes of the parameters
  if PL2==False:
    print(p,L1,L3)  # displaying the sizes of the parameters

print(np)              # total number of parameters

"""#Step 10: Building the dataset"""

d5 = df.sample(300)

d5

# Transformando em uma única string
docs_join = '\n'.join(d5['text'])

docs_join

with open('/content/drive/MyDrive/projetos/dados/pos-bumbabert/temp_texto_concatenado2.txt', 'w') as arquivo:
    arquivo.write(docs_join)

print("Texto salvo com sucesso no arquivo 'temp_texto_concatenado2.txt'")

# Commented out IPython magic to ensure Python compatibility.

from transformers import LineByLineTextDataset
 
dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path='/content/drive/MyDrive/projetos/dados/pos-bumbabert/temp_texto_concatenado2.txt', #
    block_size=128,
)

"""#Step 11: Defining a data collator"""

from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

"""#Step 12: Initializing the trainer"""

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="/content/drive/MyDrive/projetos/bumbabert/drRoBerta/model4",
    overwrite_output_dir=True,
    num_train_epochs=60,
    per_device_train_batch_size=32,
    save_steps=100,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

"""#Step 13: Pretraining the model"""

# Commented out IPython magic to ensure Python compatibility.
# %%time
trainer.train()

"""#Step 14: Saving the final model (+tokenizer + config) to disk

> Adicionar aspas


"""

trainer.save_model("/content/drive/MyDrive/projetos/bumbabert/drRoBerta/model4")

"""#Step 15: Language modeling with FillMaskPipeline"""

from transformers import pipeline

fill_mask = pipeline(
    "fill-mask",
    model="/content/drive/MyDrive/projetos/bumbabert/drRoBerta/model4",
    tokenizer="/content/drive/MyDrive/projetos/bumbabert/drRoBerta/model1"
)

fill_mask("Corte Superior de <mask>, a fim de evitar o deslocamentoda competência da Justiça Federal para a Estadual, ou vice-versa, apósdecorrida toda a instrução processual, sufragou entendimento segundo o quala competência é definida, ab initio, em razão do pedido e da causa de pedirpresentes na peça vestibular, e não por sua procedência ou improcedência,legitimidade ou ilegitimidade das partes, ou qualquer outro juízo a respeitoda própria demanda.4. Incompetência da Justiça Federal para julgar a presente demanda que se reconhece.")