from PIL import Image
import os
import numpy as np

def criar_diretorio_dataset():
    base_dir = "../Image/dataset"
    cores = ["red", "green", "blue", "yellow"]
    
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        print(f"✓ Diretório principal criado: {base_dir}")
    
    # Criar sub diretorios para cada cor
    for cor in cores:
        cor_dir = os.path.join(base_dir, cor)
        if not os.path.exists(cor_dir):
            os.makedirs(cor_dir)
            print(f"✓ Diretório {cor_dir} criado")
    
    return base_dir, cores

def gerar_variacoes_cor(cor_base, num_variacoes=50):
    # parametros:
    # cor_base: RGB da cor principal tipo: (255, 0, 0) para vermelho
    # num_variacoes: quantas imagens diferentes gerar

    variacoes = []
    r, g, b = cor_base
    
    for i in range(num_variacoes):
        # Adicionar pequenas variações aleatórias (-30 a +30)
        variacao_r = max(0, min(255, r + np.random.randint(-30, 31)))
        variacao_g = max(0, min(255, g + np.random.randint(-30, 31)))
        variacao_b = max(0, min(255, b + np.random.randint(-30, 31)))
        
        variacoes.append((variacao_r, variacao_g, variacao_b))
    
    return variacoes

# Criar a imagem colorida no tamanho 28 x 28
def criar_imagem_colorida(cor_rgb, tamanho=(28, 28)):
    imagem = Image.new('RGB', tamanho, cor_rgb)
    return imagem

def salvar_dataset_completo():
    print("INICIANDO CRIAÇÃO DO DATASET...")
    print("=" * 50)
    
    # Criar estrutura de diretórios
    base_dir, nomes_cores = criar_diretorio_dataset()
    
    # Definir cores base em RGB
    cores_rgb = {
        "red": (255, 0, 0),      # Vermelho
        "green": (0, 255, 0),    # Verde  
        "blue": (0, 0, 255),     # Azul
        "yellow": (255, 255, 0)  # Amarelo
    }
    
    print("\nGERANDO IMAGENS PARA CADA COR...")
    
    for nome_cor in nomes_cores:
        print(f"\nProcessando cor: {nome_cor.upper()}")
        
        # Gerar variações da cor
        cor_base = cores_rgb[nome_cor]
        variacoes = gerar_variacoes_cor(cor_base, num_variacoes=50)
        
        # Criar e salvar cada imagem
        for i, cor_variacao in enumerate(variacoes, 1):
            # Criar imagem
            imagem = criar_imagem_colorida(cor_variacao)
            
            # Definir nome do arquivo
            nome_arquivo = f"{nome_cor}_{i:02d}.png"
            caminho_completo = os.path.join(base_dir, nome_cor, nome_arquivo)
            
            # Salvar imagem
            imagem.save(caminho_completo)
            
            if i % 10 == 0:  # Mostrar progresso a cada 10 imagens
                print(f"   ✓ {i}/50 imagens criadas...")
        
        print(f"Concluído! 50 imagens de {nome_cor} salvas.")
    
    print("\n" + "=" * 50)
    print("DATASET CRIADO COM SUCESSO!")
    print("Classes criadas: Vermelho, Verde, Azul, Amarelo")

if __name__ == "__main__":
    # Executar criação do dataset
    salvar_dataset_completo()