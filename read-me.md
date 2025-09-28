==================================================================================
# Observação: Arquivo Markdown gerado com apoio do Copilot, resumindo o processo realizado. 
==================================================================================
# Como rodar o projeto:

- 1. Crie o dataset (caso nao tenha um). Para isso execute o comando que gera as imagens no padrão, troque o diretorio para o seu PC:

cd "E:\Faculdade2Q25\ListasH1\Lista 3\Uteis\"
python criar_dataset.py

- 2. Treinar rede neural

cd "E:\Faculdade2Q25\ListasH1\Lista 3\Python"
python ReconhecedorDeCores.py

==================================================================================

# 1️. CRIAÇÃO DO DATASET
- 200 imagens criadas (50 imagens por cor)
- 4 classes: Vermelho, Verde, Azul, Amarelo
- Variações de brilho para enriquecer o dataset
- Formato 28x28 pixels

# 2️. IMPLEMENTAÇÃO DA REDE NEURAL (ReconhecedorDeCores.py)
- Arquitetura: 4 neurônios de saída
- Forward Pass: Implementação completa com ReLU e Softmax
- Backward Pass: Backpropagation com regra da cadeia
- Função de Perda: Cross-entropy loss

# 3️. TREINAMENTO E RESULTADOS
- 100 épocas de treinamento
- Acurácia final: 100% (treino e teste)
- Sem overfitting: Performance igual em treino e teste

# ARQUIVOS:
criar_dataset.py - Gera dataset de 200 imagens coloridas
ReconhecedorDeCores.py - Rede neural principal

==================================================================================
# Matemática da Rede:

Forward Pass:
- z₁ = X @ W₁ + b₁
- a₁ = ReLU(z₁) = max(0, z₁)
- z₂ = a₁ @ W₂ + b₂  
- a₂ = ReLU(z₂) = max(0, z₂)
- z₃ = a₂ @ W₃ + b₃
- a₃ = Softmax(z₃)

Backward Pass:
- dW₃ = (a₂ᵀ @ (a₃ - y_true)) / m
- dW₂ = (a₁ᵀ @ (da₂ * ReLU'(z₂))) / m
- dW₁ = (Xᵀ @ (da₁ * ReLU'(z₁))) / m

Loss Function:
- L = -Σ(y_true * log(y_pred))

- 784 neurônios de entrada: 28×28 pixels de imagem
- 64 neurônios (camada 1): Detectam padrões básicos de cor
- 32 neurônios (camada 2): Combinam padrões para reconhecer cores
- 4 neurônios de saída: Uma probabilidade para cada cor
- ReLU: Função de ativação simples e eficiente
- Softmax: Converte saídas em probabilidades
- Cross-entropy: Ideal para classificação multiclasse

# ACURÁCIA FINAL: 100%
# CLASSIFICAÇÃO POR COR:
- Vermelho: 100% correto
- Verde: 100% correto  
- Azul: 100% correto
- Amarelo: 100% correto
==================================================================================
# Por que normalizar pixels?
Pixels vão de 0-255, rede funciona melhor com 0-1
Evita gradientes muito grandes
Acelera convergência

# Por que essa arquitetura?
Entrada grande (784): Todos os pixels da imagem
Primeira camada menor (64): Extrai características básicas
Segunda camada menor ainda (32): Combina características
Saída pequena (4): Decisão final sobre a cor
Por que ReLU?
Simples: f(x) = max(0,x)
Evita vanishing gradient
Computacionalmente eficiente
Por que Softmax na saída?
Converte números em probabilidades
Soma total = 1.0
Ideal para classificação multiclasse

# RESULTADO FINAL:
A rede neural foi implementada com sucesso e demonstra 100% de acurácia na classificação das 4 cores propostas.