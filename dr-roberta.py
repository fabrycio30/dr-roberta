# -*- coding: utf-8 -*-
# Script para pré-treinamento de Bumbabert (baseado no Roberta)

import os
import torch
import pandas as pd
from pathlib import Path
from transformers import (
    RobertaConfig,
    RobertaForMaskedLM,
    RobertaTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    pipeline
)
from tokenizers import ByteLevelBPETokenizer
from accelerate import Accelerator

# Função principal para organizar o fluxo de execução
def main():
    path_data = ''
    path_model = ''
    path_tokenizer = ''
    path_logs =' '
    #1 - Carregando os dados
    data = pd.read_csv(path_data)
    df = data.sample(1000)

    # 2 - Instanciando o Acelerador
    accelerator = Accelerator()

    # 3 - Treinando o tokenizer do zero
    tokenizer = treinar_tokenizer(df)

    # 4 -  Salvando o tokenizer treinado
    token_dir = path_tokenizer
    if not os.path.exists(token_dir):
        os.makedirs(token_dir)
    tokenizer.save_model(token_dir)

    # 5 - Carregando o tokenizer treinado
    tokenizer = carregar_tokenizer(token_dir)

    # 6 - Configuração do modelo
    config = configurar_modelo()

    # 7 - Inicializando o modelo
    model = RobertaForMaskedLM(config=config)

    # 8 - Preparando o dataset e o data collator
    dataset, data_collator = preparar_dados(df, tokenizer)

    # 9 - Treinamento do modelo
    treino_modelo(model, dataset, data_collator)

    # 10 - Salvando o modelo final
    salvar_modelo_final(model, token_dir)

    # 11 - Linguagem com FillMaskPipeline
    # só depois do treinamento
    #executar_fillmask(token_dir)

# Treinando o tokenizer
def treinar_tokenizer(df):
    text = "\n".join(df['text'])
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator([text], vocab_size=52000, min_frequency=2, special_tokens=[
        "<s>", "<pad>", "</s>", "<unk>", "<mask>"
    ])
    return tokenizer

# Load do tokenizador treinado
def carregar_tokenizer(token_dir):
    tokenizer = ByteLevelBPETokenizer(
        f"{token_dir}/vocab.json",
        f"{token_dir}/merges.txt",
    )
    tokenizer._tokenizer.post_processor = BertProcessing(
        ("</s>", tokenizer.token_to_id("</s>")),
        ("<s>", tokenizer.token_to_id("<s>")),
    )
    tokenizer.enable_truncation(max_length=512)
    return tokenizer

# Configuração do modelo dr-Roberta
def configurar_modelo():
    config = RobertaConfig(
        vocab_size=30000,
        max_position_embeddings=514,
        num_attention_heads=12,
        num_hidden_layers=6,
        type_vocab_size=1
    )
    return config

# Prepararação do dataset e o do data collator
def preparar_dados(df, tokenizer):
    # Salvando texto concatenado
    docs_join = '\n'.join(df['text'])
    with open(f"{path_logs}/temp_texto_concatenado.txt", 'w') as arquivo:
        arquivo.write(docs_join)

    # load do dataset
    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=f"{path_logs}/temp_texto_concatenado.txt",
        block_size=128,
    )

    # Collator para a modelagem de linguagem mascarada
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    return dataset, data_collator

# Treinamento do modelo
def treino_modelo(model, dataset, data_collator):
    training_args = TrainingArguments(
        output_dir=f"{path_model}/model",
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

    trainer.train()

# Salvando o modelo final
def salvar_modelo_final(model, path_model):
    model.save_pretrained(f"{path_model}/final_model")
    print("Modelo salvo com sucesso.")

# Executando o pipeline fill-mask
def executar_fillmask(path_model ,token_dir):
    fill_mask = pipeline(
        "fill-mask",
        model=f"{path_model}/final_model",
        tokenizer=f"{token_dir}/"
    )
    # Teste com mascaras
    resultado = fill_mask("Corte Superior de <mask>, a fim de evitar o deslocamento da competência.")
    print(resultado)    

if __name__ == "__main__":
    main()
