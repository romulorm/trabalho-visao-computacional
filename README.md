# Trabalho da disciplina de Visão Computacional

## Preparando o ambiente

```bash
git clone https://github.com/romulorm/trabalho-visao-computacional.git
cd trabalho-visao-computacional
uv venv .venv --python 3.12
source .venv/bin/activate #caso não ative automaticamente
uv sync
```

## Testando os modelos de detecção

### Utilizando o Gradio
1) Executar o comando:
```bash
python gradio_app.py
```
Acesse a URL: http://localhost:7860

### Utilizando o Jupyter Notebook
1) Acessar o notebook teste_modelo.ipynb
2) Alterar a variável best_model_path da célula 2 para um dos modelos treinados:
- models/best1.pt (treinado com o dataset original)
- models/best2.pt (treinado com o dataset ampliado)
3) Alterar a variável img_teste entre t1.jpg e t10.jpg
