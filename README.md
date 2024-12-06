# FIAP - Faculdade de Informática e Administração Paulista

<p align="center">
<a href= "https://www.fiap.com.br/"><img src="assets/images/logo-fiap.png" alt="FIAP - Faculdade de Informática e Admnistração Paulista" border="0" width=40% height=40%></a>
</p>

<br>

# Cap 3 - Implementando algoritmos de Machine Learning com Scikit-learn

## 👨‍👩 Grupo

Grupo de número <b>52</b> formado pelos integrantes mencionados abaixo.

## 👨‍🎓 Integrantes: 
- <a href="https://www.linkedin.com/in/cirohenrique/">Ciro Henrique</a> ( <i>RM559040</i> )
- <a href="javascript:void(0)">Enyd Bentivoglio</a> ( <i>RM560234</i> )
- <a href="https://www.linkedin.com/in/marcofranzoi/">Marco Franzoi</a> ( <i>RM559468</i> )
- <a href="https://www.linkedin.com/in/rodrigo-mazuco-16749b37/">Rodrigo Mazuco</a> ( <i>RM559712</i> )

## 👩‍🏫 Professores:

### Tutor(a) 
- <a href="https://www.linkedin.com/in/lucas-gomes-moreira-15a8452a/">Lucas Gomes Moreira</a>

### Coordenador(a)
- <a href="https://www.linkedin.com/in/profandregodoi/">André Godoi</a>

## 📜 Descrição

## **Relatório de Classificação de Grãos com Machine Learning**

### **Contexto do Problema**

A classificação de grãos de trigo (Kama, Rosa e Canadian) realizada manualmente em cooperativas agrícolas é um processo demorado e sujeito a erros humanos. O objetivo deste projeto foi desenvolver um modelo automatizado de classificação utilizando aprendizado de máquina, garantindo maior precisão, eficiência e consistência.

### **Modelos Avaliados**

Três modelos de aprendizado de máquina foram otimizados e avaliados: SVM, Random Forest e KNN.

### **1. Suporte a Vetores (SVM)**

- **Melhores Parâmetros**: `{'C': 1, 'kernel': 'linear', 'degree': 2, 'gamma': 'scale'}`.
- **Desempenho**:
    - **Acurácia Geral**: 93%.
    - **Classe 1 (Kama)**: Precision = 0.87, Recall = 0.87.
    - **Classe 2 (Rosa)**: Precision = 1.00, Recall = 1.00.
    - **Classe 3 (Canadian)**: Precision = 0.90, Recall = 0.90.

### **2. Random Forest**

- **Melhores Parâmetros**: `{'n_estimators': 50, 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 2}`.
- **Desempenho**:
    - **Acurácia Geral**: 92%.
    - **Classe 1 (Kama)**: Precision = 0.86, Recall = 0.80.
    - **Classe 2 (Rosa)**: Precision = 1.00, Recall = 0.96.
    - **Classe 3 (Canadian)**: Precision = 0.86, Recall = 0.95.

### **3. K-Nearest Neighbors (KNN)**

- **Melhores Parâmetros**: `{'n_neighbors': 5, 'weights': 'distance'}`.
- **Desempenho**:
    - **Acurácia Geral**: 92%.
    - **Classe 1 (Kama)**: Precision = 0.86, Recall = 0.80.
    - **Classe 2 (Rosa)**: Precision = 1.00, Recall = 1.00.
    - **Classe 3 (Canadian)**: Precision = 0.86, Recall = 0.90.

### **Insights Relevantes**

### **Separação entre Classes**

- **Classe 2 (Rosa)** apresentou a melhor separação, com precisão e recall perfeitos em todos os modelos.
- **Classes 1 (Kama) e 3 (Canadian)** mostraram certa sobreposição, indicando similaridade em algumas características.

### **Importância das Variáveis**

- No Random Forest, **Comprimento do Sulco do Núcleo**, **Área**, e **Perímetro** foram identificados como as variáveis mais importantes para a classificação.

### **Modelo Recomendado**

- **SVM com kernel linear** é o modelo recomendado para implementação devido à sua maior acurácia (93%) e desempenho equilibrado.
- Random Forest é uma alternativa interessante pela interpretabilidade das variáveis.

### **Conclusão**

Os modelos desenvolvidos são altamente eficazes, com acurácia acima de 90%, superando significativamente a classificação manual. A automação da classificação de grãos com aprendizado de máquina proporciona:

- **Maior precisão**: Minimiza erros humanos.
- **Consistência**: Garante padrões homogêneos na classificação.
- **Eficiência**: Reduz o tempo necessário para classificar grandes volumes.

Com base nos resultados, recomenda-se a implementação do SVM para o ambiente de produção e a contínua avaliação com novos dados para garantir a robustez do modelo.

## **Relatório de Classificação de Grãos com Machine Learning**

### **Contexto do Problema**

A classificação de grãos de trigo (Kama, Rosa e Canadian) realizada manualmente em cooperativas agrícolas é um processo demorado e sujeito a erros humanos. O objetivo deste projeto foi desenvolver um modelo automatizado de classificação utilizando aprendizado de máquina, garantindo maior precisão, eficiência e consistência.

### **Modelos Avaliados**

Três modelos de aprendizado de máquina foram otimizados e avaliados: SVM, Random Forest e KNN.

### **1. Suporte a Vetores (SVM)**

- **Melhores Parâmetros**: `{'C': 1, 'kernel': 'linear', 'degree': 2, 'gamma': 'scale'}.~`
- **Desempenho**:
    - **Acurácia Geral**: 93%.
    - **Classe 1 (Kama)**: Precision = 0.87, Recall = 0.87.
    - **Classe 2 (Rosa)**: Precision = 1.00, Recall = 1.00.
    - **Classe 3 (Canadian)**: Precision = 0.90, Recall = 0.90.
- **Matriz de Confusão**:
    `[[13  0  2]
    [ 0 25  0]
    [ 2  0 18]]`


### **2. Random Forest**

- **Melhores Parâmetros**: `{'n_estimators': 50, 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 2}`
- **Desempenho**:
    - **Acurácia Geral**: 92%.
    - **Classe 1 (Kama)**: Precision = 0.86, Recall = 0.80.
    - **Classe 2 (Rosa)**: Precision = 1.00, Recall = 0.96.
    - **Classe 3 (Canadian)**: Precision = 0.86, Recall = 0.95.
- **Matriz de Confusão**:
    `[[12  0  3]
    [ 1 24  0]
    [ 1  0 19]]`

### **3. K-Nearest Neighbors (KNN)**

- **Melhores Parâmetros**: `{'n_neighbors': 5, 'weights': 'distance'}`
- **Desempenho**:
    - **Acurácia Geral**: 92%.
    - **Classe 1 (Kama)**: Precision = 0.86, Recall = 0.80.
    - **Classe 2 (Rosa)**: Precision = 1.00, Recall = 1.00.
    - **Classe 3 (Canadian)**: Precision = 0.86, Recall = 0.90.
- **Matriz de Confusão**:
    `[[12  0  3]
    [ 0 25  0]
    [ 2  0 18]]`

### **Insights Relevantes**

### **Separação entre Classes**

- **Classe 2 (Rosa)** apresentou a melhor separação, com precisão e recall perfeitos em todos os modelos.
- **Classes 1 (Kama) e 3 (Canadian)** mostraram certa sobreposição, indicando similaridade em algumas características.

### **Importância das Variáveis**

- No Random Forest, **Comprimento do Sulco do Núcleo**, **Área**, e **Perímetro** foram identificados como as variáveis mais importantes para a classificação.

### **Modelo Recomendado**

- **SVM com kernel linear** é o modelo recomendado para implementação devido à sua maior acurácia (93%) e desempenho equilibrado. 
- Random Forest é uma alternativa interessante pela interpretabilidade das variáveis.

### **Conclusão**

Os modelos desenvolvidos são altamente eficazes, com acurácia acima de 90%, superando significativamente a classificação manual. A automação da classificação de grãos com aprendizado de máquina proporciona:

- **Maior precisão**: Minimiza erros humanos.
- **Consistência**: Garante padrões homogêneos na classificação.
- **Eficiência**: Reduz o tempo necessário para classificar grandes volumes.

Com base nos resultados, recomenda-se a implementação do SVM para o ambiente de produção e a contínua avaliação com novos dados para garantir a robustez do modelo.

## 📁 Estrutura de pastas

Dentre os arquivos e pastas presentes na raiz do projeto, definem-se:

1. <b>assets</b>: Diretório para armazenamento de arquivos complementares da estrutura do projeto.
    - Diretório "images": Diretório para armazenamento de imagens.

2. <b>config</b>: Diretório para armazenamento de arquivos em formato <i>json</i> contendo configurações.

3. <b>document</b>: Diretório para armazenamento de documentos relacionados ao projeto.

4. <b>scripts</b>: Diretório para armazenamento de scripts.

5. <b>src</b>: Diretório para armazenamento de código fonte do projeto.

6. <b>tests</b>: Diretório para armazenamento de resultados de testes.
	- Diretório "images": Diretório para armazenamento de imagens relacionadas aos testes efetuados.
        - Diretório "graphics": Diretório para armazenamento das imagens referentes aos gráficos gerados.

7. <b>README.md</b>: Documentação do projeto em formato markdown.

## 🔧 Como executar o código

Os arquivos do projeto estão em seu formato original no diretório "src".

## 🗃 Histórico de lançamentos

* 1.0.0 - 06/12/2024

## 📋 Licença

Desenvolvido pelo Grupo 52 para o projeto da fase 4 (<i>Cap 3 - Implementando algoritmos de Machine Learning com Scikit-learn</i>) da <a rel="cc:attributionURL dct:creator" property="cc:attributionName" href="https://fiap.com.br">Fiap</a>. Está licenciado sobre <a href="http://creativecommons.org/licenses/by/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">Attribution 4.0 International</a>.</p>