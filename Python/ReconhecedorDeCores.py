import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import random

# Esta classe implementa todas as operações matemáticas necessárias
class RedeNeuralCores:

        # CONSTRUTOR: Inicializa a rede neural
        # PARÂMETROS:
        # tamanho_entrada: 784 (28x28 pixels da imagem)
        # camada1: 64 neurônios na primeira camada oculta
        # camada2: 32 neurônios na segunda camada oculta
        # num_classes: 4 classes de cores
        # taxa_aprendizado: velocidade de aprendizado

    def __init__(self, tamanho_entrada=784, camada1=64, camada2=32, num_classes=4, taxa_aprendizado=0.01):
        self.taxa_aprendizado = taxa_aprendizado
        
        # INICIALIZAÇÃO DOS PESOS (WEIGHTS)
        # Pesos da camada de entrada para primeira camada oculta
        self.W1 = np.random.randn(tamanho_entrada, camada1) * np.sqrt(2.0 / tamanho_entrada)
        self.b1 = np.zeros((1, camada1))  # Bias da primeira camada
        
        # Pesos da primeira para segunda camada oculta
        self.W2 = np.random.randn(camada1, camada2) * np.sqrt(2.0 / camada1)
        self.b2 = np.zeros((1, camada2))  # Bias da segunda camada
        
        # Pesos da segunda camada oculta para saída
        self.W3 = np.random.randn(camada2, num_classes) * np.sqrt(2.0 / camada2)
        self.b3 = np.zeros((1, num_classes))  # Bias da camada de saída

        print(f"Rede inicializada: {tamanho_entrada} -> {camada1} -> {camada2} -> {num_classes}")

# FUNÇÃO DE ATIVAÇÃO ReLU (Rectified Linear Unit)
# EXPLICAÇÃO MATEMÁTICA:
# - ReLU(x) = max(0, x)
# - Se x > 0, retorna x
# - Se x ≤ 0, retorna 0
    def relu(self, x):
        return np.maximum(0, x)
    
# DERIVADA DA ReLU para backpropagation 
# EXPLICAÇÃO MATEMÁTICA:
# - Se x > 0, derivada = 1
# - Se x ≤ 0, derivada = 0
# - Necessária para calcular gradientes no treinamento
    
    def relu_derivada(self, x):
        return (x > 0).astype(float)

# FUNÇÃO DE ATIVAÇÃO SOFTMAX para camada de saída
        
# EXPLICAÇÃO MATEMÁTICA:
# - Converte números em probabilidades que somam 1
# - softmax(x_i) = exp(x_i) / sum(exp(x_j) for all j)
# - Ideal para classificação multiclasse
        
# EXEMPLO:
# Entrada: [2.0, 1.0, 0.1, 3.0]
# - Saída: [0.24, 0.09, 0.03, 0.64] (soma = 1.0)
    def softmax(self, x):
        # Subtraímos o máximo para estabilidade numérica (evita overflow)
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward_pass(self, X):
        """
        FORWARD PASS: Propaga dados da entrada até a saída
        
        EXPLICAÇÃO PASSO A PASSO:
        1. Entrada X (imagens normalizadas) entra na rede
        2. Primeira camada: X * W1 + b1, depois ReLU
        3. Segunda camada: resultado * W2 + b2, depois ReLU  
        4. Saída: resultado * W3 + b3, depois Softmax
        
        MATEMÁTICA:
        - z = X @ W + b (multiplicação matricial + bias)
        - a = activation_function(z)
        """
        # Primeira camada oculta
        self.z1 = X @ self.W1 + self.b1  # Combinação linear
        self.a1 = self.relu(self.z1)     # Ativação ReLU
        
        # Segunda camada oculta
        self.z2 = self.a1 @ self.W2 + self.b2  # Combinação linear
        self.a2 = self.relu(self.z2)           # Ativação ReLU
        
        # Camada de saída
        self.z3 = self.a2 @ self.W3 + self.b3  # Combinação linear
        self.a3 = self.softmax(self.z3)        # Ativação Softmax
        
        return self.a3
    
    def calcular_perda(self, y_pred, y_true):
        """
        FUNÇÃO DE PERDA: Cross-Entropy Loss
        
        EXPLICAÇÃO MATEMÁTICA:
        - Loss = -sum(y_true * log(y_pred))
        - Penaliza previsões incorretas exponencialmente
        - Ideal para classificação multiclasse
        
        EXEMPLO:
        - Se classe correta tem probabilidade 0.9 → perda baixa
        - Se classe correta tem probabilidade 0.1 → perda alta
        """
        # Evitar log(0) adicionando pequeno epsilon
        epsilon = 1e-15
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Cross-entropy loss
        return -np.mean(np.sum(y_true * np.log(y_pred_clipped), axis=1))
    
    def backward_pass(self, X, y_true):
        """
        BACKWARD PASS: Backpropagation para calcular gradientes
        
        EXPLICAÇÃO DETALHADA:
        - Calcula como cada peso contribui para o erro
        - Usa regra da cadeia para propagar gradientes
        - Atualiza pesos na direção que reduz o erro
        
        MATEMÁTICA:
        - dW = dL/dW (derivada da perda em relação aos pesos)
        - W_novo = W_antigo - taxa_aprendizado * dW
        """
        m = X.shape[0]  # número de amostras
        
        # Gradiente da camada de saída
        dz3 = self.a3 - y_true  # Derivada da softmax com cross-entropy
        dW3 = (self.a2.T @ dz3) / m
        db3 = np.sum(dz3, axis=0, keepdims=True) / m
        
        # Gradiente da segunda camada oculta
        da2 = dz3 @ self.W3.T
        dz2 = da2 * self.relu_derivada(self.z2)
        dW2 = (self.a1.T @ dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        # Gradiente da primeira camada oculta
        da1 = dz2 @ self.W2.T
        dz1 = da1 * self.relu_derivada(self.z1)
        dW1 = (X.T @ dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        # Atualizar pesos e bias (Gradient Descent)
        self.W3 -= self.taxa_aprendizado * dW3
        self.b3 -= self.taxa_aprendizado * db3
        self.W2 -= self.taxa_aprendizado * dW2
        self.b2 -= self.taxa_aprendizado * db2
        self.W1 -= self.taxa_aprendizado * dW1
        self.b1 -= self.taxa_aprendizado * db1
    
    def prever(self, X):
        """
        FUNÇÃO DE PREDIÇÃO: Classifica novas imagens
        
        EXPLICAÇÃO:
        - Executa forward pass
        - Retorna classe com maior probabilidade
        - Usado após o treinamento para testar a rede
        """
        probabilidades = self.forward_pass(X)
        return np.argmax(probabilidades, axis=1)
    
    def acuracia(self, X, y_true):
        """
        FUNÇÃO DE AVALIAÇÃO: Calcula precisão da rede
        
        EXPLICAÇÃO:
        - Compara previsões com rótulos verdadeiros
        - Retorna percentual de acertos
        - Métrica principal para avaliar performance
        """
        predicoes = self.prever(X)
        y_true_classes = np.argmax(y_true, axis=1)
        return np.mean(predicoes == y_true_classes)

class CarregadorDados:
    """
    CLASSE AUXILIAR: Carrega e processa dataset de imagens
    
    EXPLICAÇÃO:
    - Lê imagens do diretório dataset
    - Converte para arrays numpy
    - Normaliza pixels para [0,1]
    - Cria rótulos one-hot encoding
    """
    
    def __init__(self, caminho_dataset="../Image/dataset"):
        self.caminho_dataset = caminho_dataset
        self.classes = ["red", "green", "blue", "yellow"]
        self.mapeamento_classes = {classe: i for i, classe in enumerate(self.classes)}
        
    def carregar_imagem(self, caminho_imagem):
        """
        FUNÇÃO: Carrega e processa uma imagem
        
        EXPLICAÇÃO DETALHADA:
        1. Abre imagem com PIL
        2. Converte para RGB (caso seja diferente)
        3. Transforma em array numpy
        4. Normaliza pixels de [0,255] para [0,1]
        5. Achata de (28,28,3) para (2352,) - vetor 1D
        
        POR QUE NORMALIZAR?
        - Pixels vão de 0 a 255 (inteiros)
        - Redes neurais funcionam melhor com valores [0,1]
        - Evita problemas de gradientes muito grandes
        """
        try:
            # Carregar e converter imagem
            img = Image.open(caminho_imagem).convert('RGB')
            img_array = np.array(img)
            
            # Normalizar pixels para [0,1]
            img_normalizada = img_array.astype(np.float32) / 255.0
            
            # Achatar para vetor 1D
            img_vetor = img_normalizada.flatten()
            
            return img_vetor
        except Exception as e:
            print(f"Erro ao carregar {caminho_imagem}: {e}")
            return None
    
    def carregar_dataset_completo(self):
        """
        FUNÇÃO PRINCIPAL: Carrega todo o dataset
        
        EXPLICAÇÃO DO PROCESSO:
        1. Varre todas as pastas de cores
        2. Carrega cada imagem
        3. Cria arrays X (dados) e y (rótulos)
        4. Embaralha dados para melhor treinamento
        
        RETORNA:
        - X: array (n_amostras, 2352) com pixels normalizados
        - y: array (n_amostras, 4) com rótulos one-hot
        """
        print("📁 Carregando dataset de imagens...")
        
        X = []  # Lista para armazenar imagens
        y = []  # Lista para armazenar rótulos
        
        for classe in self.classes:
            print(f"   📸 Carregando imagens de {classe}...")
            caminho_classe = os.path.join(self.caminho_dataset, classe)
            
            if not os.path.exists(caminho_classe):
                print(f"   ⚠️ Diretório não encontrado: {caminho_classe}")
                continue
            
            # Listar todas as imagens da classe
            arquivos_imagem = [f for f in os.listdir(caminho_classe) if f.endswith('.png')]
            
            for arquivo in arquivos_imagem:
                caminho_completo = os.path.join(caminho_classe, arquivo)
                img_vetor = self.carregar_imagem(caminho_completo)
                
                if img_vetor is not None:
                    X.append(img_vetor)
                    # Criar rótulo one-hot
                    rotulo = [0] * len(self.classes)
                    rotulo[self.mapeamento_classes[classe]] = 1
                    y.append(rotulo)
            
            print(f"   ✓ {len(arquivos_imagem)} imagens de {classe} carregadas")
        
        # Converter listas para arrays numpy
        X = np.array(X)
        y = np.array(y)
        
        # Embaralhar dados para melhor treinamento
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        print(f"\n✅ Dataset carregado com sucesso!")
        print(f"   📊 Formato dos dados: {X.shape}")
        print(f"   🏷️  Formato dos rótulos: {y.shape}")
        print(f"   🎯 Classes: {self.classes}")
        
        return X, y
    
    def dividir_treino_teste(self, X, y, proporcao_teste=0.2):
        """
        FUNÇÃO: Divide dataset em treino e teste
        
        EXPLICAÇÃO:
        - 80% dos dados para treinamento
        - 20% dos dados para teste
        - Importante para avaliar generalização da rede
        
        POR QUE DIVIDIR?
        - Treino: rede aprende com estes dados
        - Teste: avalia se rede funciona em dados nunca vistos
        - Evita overfitting (decorar ao invés de aprender)
        """
        n_teste = int(len(X) * proporcao_teste)
        
        X_teste = X[:n_teste]
        y_teste = y[:n_teste]
        X_treino = X[n_teste:]
        y_treino = y[n_teste:]
        
        print(f"📊 Divisão dos dados:")
        print(f"   🎯 Treino: {len(X_treino)} amostras")
        print(f"   🧪 Teste: {len(X_teste)} amostras")
        
        return X_treino, y_treino, X_teste, y_teste

def treinar_rede_neural():
    """
    FUNÇÃO PRINCIPAL DE TREINAMENTO
    
    EXPLICAÇÃO DO PROCESSO COMPLETO:
    1. Carrega dataset de imagens
    2. Divide em treino/teste
    3. Cria e inicializa rede neural
    4. Executa loop de treinamento (épocas)
    5. Avalia performance final
    """
    print("🚀 INICIANDO TREINAMENTO DA REDE NEURAL")
    print("=" * 60)
    
    # PASSO 1: Carregar dados
    carregador = CarregadorDados()
    X, y = carregador.carregar_dataset_completo()
    
    # PASSO 2: Dividir treino/teste
    X_treino, y_treino, X_teste, y_teste = carregador.dividir_treino_teste(X, y)
    
    # PASSO 3: Criar rede neural
    print("\n🧠 Criando rede neural...")
    rede = RedeNeuralCores(
        tamanho_entrada=X.shape[1],  # Tamanho do vetor de pixels
        camada1=64,                  # Primeira camada oculta
        camada2=32,                  # Segunda camada oculta
        num_classes=4,               # 4 cores
        taxa_aprendizado=0.01        # Learning rate
    )
    
    # PASSO 4: Loop de treinamento
    print(f"\n🔄 Iniciando treinamento por {EPOCHS} épocas...")
    print("-" * 60)
    
    historico_perda = []
    historico_acuracia = []
    
    for epoca in range(EPOCHS):
        # Forward pass
        y_pred = rede.forward_pass(X_treino)
        
        # Calcular perda
        perda = rede.calcular_perda(y_pred, y_treino)
        
        # Backward pass (atualizar pesos)
        rede.backward_pass(X_treino, y_treino)
        
        # Calcular acurácia
        acuracia_treino = rede.acuracia(X_treino, y_treino)
        
        # Armazenar métricas
        historico_perda.append(perda)
        historico_acuracia.append(acuracia_treino)
        
        # Mostrar progresso a cada 10 épocas
        if (epoca + 1) % 10 == 0:
            print(f"Época {epoca+1:3d}/{EPOCHS} | "
                  f"Perda: {perda:.4f} | "
                  f"Acurácia Treino: {acuracia_treino:.3f}")
    
    # PASSO 5: Avaliação final
    print("\n" + "=" * 60)
    print("📊 AVALIAÇÃO FINAL DA REDE")
    print("=" * 60)
    
    acuracia_treino_final = rede.acuracia(X_treino, y_treino)
    acuracia_teste_final = rede.acuracia(X_teste, y_teste)
    
    print(f"🎯 Acurácia no Treino: {acuracia_treino_final:.3f} ({acuracia_treino_final*100:.1f}%)")
    print(f"🧪 Acurácia no Teste:  {acuracia_teste_final:.3f} ({acuracia_teste_final*100:.1f}%)")
    
    # Analisar resultado
    if acuracia_teste_final > 0.9:
        print("🏆 EXCELENTE! Rede aprendeu muito bem!")
    elif acuracia_teste_final > 0.7:
        print("✅ BOM! Rede tem performance satisfatória!")
    else:
        print("⚠️ Rede precisa de mais treinamento ou ajustes...")
    
    return rede, carregador, historico_perda, historico_acuracia

def testar_com_imagem_nova(rede, carregador):
    """
    FUNÇÃO: Testa rede com uma nova imagem
    
    EXPLICAÇÃO:
    - Permite testar a rede com imagens que ela nunca viu
    - Útil para validar se realmente aprendeu
    - Mostra as probabilidades de cada classe
    """
    print("\n" + "=" * 60)
    print("🔍 TESTE COM NOVA IMAGEM")
    print("=" * 60)
    
    # Criar uma imagem de teste simples
    print("Criando imagem de teste (azul)...")
    
    # Imagem azul 28x28
    img_teste = np.zeros((28, 28, 3))
    img_teste[:, :, 2] = 1.0  # Canal azul = 1.0
    
    # Processar imagem como a rede espera
    img_vetor = img_teste.flatten()
    img_batch = img_vetor.reshape(1, -1)  # Batch de 1 imagem
    
    # Fazer previsão
    probabilidades = rede.forward_pass(img_batch)[0]
    classe_prevista = np.argmax(probabilidades)
    
    print(f"\n📊 RESULTADO DA PREVISÃO:")
    print(f"   🎯 Classe prevista: {carregador.classes[classe_prevista]}")
    print(f"   📈 Confiança: {probabilidades[classe_prevista]:.3f}")
    
    print(f"\n📋 Probabilidades por classe:")
    for i, classe in enumerate(carregador.classes):
        print(f"   {classe:8s}: {probabilidades[i]:.3f}")

# PARÂMETROS DE TREINAMENTO
EPOCHS = 100  # Número de épocas de treinamento

if __name__ == "__main__":
    """
    EXECUÇÃO PRINCIPAL DO PROGRAMA
    
    FLUXO COMPLETO:
    1. Treina a rede neural
    2. Testa com nova imagem
    3. Mostra resultados
    """
    # Treinar rede
    rede, carregador, historico_perda, historico_acuracia = treinar_rede_neural()
    
    # Testar com nova imagem
    testar_com_imagem_nova(rede, carregador)
    
    print("\n🎉 PROGRAMA CONCLUÍDO COM SUCESSO!")
    print("💡 Dica: Experimente ajustar parâmetros para melhorar a performance!")