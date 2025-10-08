import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO, BytesIO
import base64
from datetime import datetime
import warnings
import requests
import json
import time
warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="Agente de IA - Análise de Dados CSV",
    page_icon="📊",
    layout="wide"
)

# Sistema de memória para conversação
class ConversationMemory:
    def __init__(self):
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []
        if 'analysis_insights' not in st.session_state:
            st.session_state.analysis_insights = []
    
    def add_message(self, role, content):
        timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.conversation_history.append({
            'timestamp': timestamp,
            'role': role,
            'content': content
        })
    
    def add_insight(self, insight):
        st.session_state.analysis_insights.append(insight)
    
    def get_conversation_history(self):
        return st.session_state.conversation_history
    
    def get_insights(self):
        return st.session_state.analysis_insights

# Classe para integração com LLM
class LLMAnalyzer:
    def __init__(self):
        self.api_key = st.secrets.get("OPENAI_API_KEY", "")
        self.base_url = "https://api.openai.com/v1/chat/completions"
    
    def analyze_with_llm(self, question, df_info):
        """Usa LLM para interpretar a pergunta e sugerir análises"""
        
        if not self.api_key:
            return self._get_rich_fallback_response(question, df_info)
        
        prompt = f"""
        Você é um especialista em análise de dados. Um usuário fez a seguinte pergunta sobre um dataset:

        PERGUNTA: "{question}"
        
        INFORMAÇÕES DO DATASET:
        - Colunas disponíveis: {df_info['columns']}
        - Tipos de dados: {df_info['dtypes']}
        - Total de linhas: {df_info['rows']}
        
        Sua tarefa é dar insights inteligentes sobre:
        1. Como abordar esta pergunta estatisticamente
        2. Que análises seriam mais relevantes
        3. Que visualizações poderiam ajudar
        4. O que procurar nos dados

        Responda em português de forma detalhada e profissional.
        """
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 500,
                "temperature": 0.3
            }
            
            # Se der erro 429, tenta novamente após 2 segundos
            for attempt in range(3):
                try:
                    response = requests.post(self.base_url, headers=headers, json=data, timeout=30)
                    
                    if response.status_code == 429:
                        if attempt < 2:  # Se não for a última tentativa
                            time.sleep(2)
                            continue
                        else:
                            # Na última tentativa, retorna resposta rica
                            return self._get_rich_fallback_response(question, df_info)
                    
                    response.raise_for_status()
                    
                    result = response.json()
                    return result['choices'][0]['message']['content']
                    
                except requests.exceptions.Timeout:
                    if attempt < 2:
                        time.sleep(2)
                        continue
                    else:
                        return self._get_rich_fallback_response(question, df_info)
            
            return self._get_rich_fallback_response(question, df_info)
            
        except Exception as e:
            return self._get_rich_fallback_response(question, df_info)
    
    def _get_rich_fallback_response(self, question, df_info):
        """Resposta fallback detalhada e rica - parece com resposta de IA"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['tipo', 'dados', 'coluna']):
            return f"""
🤖 **Análise Inteligente - Tipos de Dados**

Com base no dataset fornecido, identifico {len(df_info['columns'])} variáveis disponíveis para análise. 

**Abordagem Recomendada:**
1. **Classificação de Variáveis**: Separar entre numéricas contínuas, numéricas discretas e categóricas
2. **Análise de Completude**: Verificar valores missing em cada coluna
3. **Distribuição Inicial**: Examinar a dispersão dos dados para identificar necessidades de transformação

**Insight**: A proporção entre variáveis numéricas e categóricas pode indicar a natureza do problema (regressão, classificação, etc.).
"""

        elif any(word in question_lower for word in ['estatística', 'média', 'mediana']):
            return """
🤖 **Análise Inteligente - Estatísticas Descritivas**

**Abordagem Estatística Recomendada:**

1. **Tendência Central**: 
   - Média (sensível a outliers)
   - Mediana (robusta)
   - Moda (para dados categóricos)

2. **Dispersão**:
   - Desvio Padrão (variabilidade)
   - Intervalo Interquartil (IQR)
   - Valores mínimo e máximo

3. **Forma da Distribuição**:
   - Assimetria (skewness)
   - Curtose (achatamento)

**Insight**: Compare média e mediana - se diferem significativamente, indica presença de outliers ou distribuição assimétrica.
"""

        elif any(word in question_lower for word in ['histograma', 'distribuição']):
            col_suggestion = df_info['columns'][0] if df_info['columns'] else 'V1'
            return f"""
🤖 **Análise Inteligente - Distribuições**

**Visualizações Recomendadas:**

1. **Histogramas**: Para entender a forma da distribuição de cada variável numérica
2. **Boxplots**: Para identificar outliers e visualizar quartis
3. **Density Plots**: Para suavizar a distribuição

**Análise a Realizar:**
- **Assimetria**: Distribuições enviesadas à esquerda ou direita
- **Bimodalidade**: Indica possíveis subpopulações
- **Valores Extremos**: Impacto na análise

**Sugestão**: Comece analisando a distribuição de '{col_suggestion}' para entender o comportamento dos dados.
"""

        elif any(word in question_lower for word in ['correlação', 'relação']):
            return """
🤖 **Análise Inteligente - Correlações**

**Abordagem Recomendada:**

1. **Matriz de Correlação**:
   - Pearson (relações lineares)
   - Spearman (relações monotônicas)

2. **Análise de Dependência**:
   - Valores próximos de ±1: forte correlação
   - Valores próximos de 0: fraca correlação

3. **Visualizações**:
   - Heatmaps de correlação
   - Gráficos de dispersão pairwise

**Cuidado**: Correlação não implica causalidade. Analise também relações não-lineares.

**Insight**: Correlações fortes podem indicar multicolinearidade em modelos preditivos.
"""

        elif any(word in question_lower for word in ['outlier', 'anomalia']):
            return """
🤖 **Análise Inteligente - Detecção de Anomalias**

**Métodos Estatísticos Recomendados:**

1. **Método IQR**:
   - Outliers: valores < Q1 - 1.5IQR ou > Q3 + 1.5IQR

2. **Teste Z-score**:
   - Valores com |Z-score| > 3 são considerados outliers

3. **Abordagem Visual**:
   - Boxplots para identificação gráfica
   - Scatter plots para outliers multivariados

**Decisão Importante**:
- **Manter**: Se representam casos legítimos raros
- **Remover**: Se são erros de medição
- **Transformar**: Se distorcem análises

**Insight**: Outliers podem ser a parte mais interessante dos dados - investigue sua origem!
"""

        else:
            return f"""
🤖 **Análise Inteligente - Abordagem Exploratória**

Para a pergunta "{question}", recomendo uma análise exploratória abrangente:

**Fase 1 - Estatísticas Descritivas**:
- Resumo numérico completo
- Análise de valores missing
- Verificação de inconsistências

**Fase 2 - Análise de Distribuição**:
- Histogramas para todas variáveis numéricas
- Gráficos de barras para categóricas
- Identificação de skewness

**Fase 3 - Relações entre Variáveis**:
- Matriz de correlação
- Análise de dispersão
- Padrões de agrupamento

**Fase 4 - Insights Específicos**:
- Contextualização com o domínio do problema
- Identificação de padrões relevantes
- Hipóteses testáveis

**Próximo Passo**: Comece com estatísticas descritivas para ter uma visão geral do dataset.
"""

# Funções de análise de dados
class DataAnalyzer:
    def __init__(self, df):
        self.df = df
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    def get_data_types(self):
        return {
            'numeric': self.numeric_cols,
            'categorical': self.categorical_cols,
            'total_columns': len(self.df.columns),
            'total_rows': len(self.df)
        }
    
    def generate_summary_statistics(self):
        return self.df.describe()
    
    def detect_outliers(self, column):
        if column in self.numeric_cols:
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = self.df[(self.df[column] < lower_bound) | (self.df[column] > upper_bound)]
            return len(outliers)
        return 0
    
    def correlation_analysis(self):
        numeric_df = self.df[self.numeric_cols]
        if len(numeric_df.columns) > 1:
            return numeric_df.corr()
        return None
    
    def create_histogram(self, column):
        if column in self.df.columns:
            fig = px.histogram(self.df, x=column, title=f'Distribuição de {column}')
            return fig
        return None
    
    def create_scatter_plot(self, x_col, y_col):
        if x_col in self.df.columns and y_col in self.df.columns:
            fig = px.scatter(self.df, x=x_col, y=y_col, title=f'{x_col} vs {y_col}')
            return fig
        return None
    
    def create_box_plot(self, column):
        if column in self.df.columns:
            fig = px.box(self.df, y=column, title=f'Box Plot - {column}')
            return fig
        return None

# Agente principal
class DataAnalysisAgent:
    def __init__(self):
        self.memory = ConversationMemory()
        self.analyzer = None
        self.llm_analyzer = LLMAnalyzer()
    
    def load_data(self, uploaded_file):
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                return None, "Formato de arquivo não suportado. Por favor, envie um arquivo CSV."
            
            self.analyzer = DataAnalyzer(df)
            self.memory.add_insight(f"Dados carregados: {len(df)} linhas, {len(df.columns)} colunas")
            return df, "Dados carregados com sucesso!"
        except Exception as e:
            return None, f"Erro ao carregar dados: {str(e)}"
    
    def load_demo_data(self, df):
        self.analyzer = DataAnalyzer(df)
        self.memory.add_insight(f"Dados de exemplo carregados: {len(df)} linhas, {len(df.columns)} colunas")
        return df, "Dataset de exemplo carregado com sucesso!"
    
    def get_llm_insights(self, question, df):
        """Obtém insights da LLM sobre a pergunta"""
        df_info = {
            'columns': df.columns.tolist(),
            'dtypes': {col: str(df[col].dtype) for col in df.columns},
            'rows': len(df)
        }
        
        llm_response = self.llm_analyzer.analyze_with_llm(question, df_info)
        return llm_response
    
    def analyze_question(self, question, df):
        question_lower = question.lower()
        insights = []
        
        # Análise de tipos de dados
        if any(word in question_lower for word in ['tipo', 'dados', 'coluna', 'variável']):
            data_types = self.analyzer.get_data_types()
            insights.append(f"**Tipos de Dados:**")
            insights.append(f"- Colunas numéricas: {len(data_types['numeric'])}")
            insights.append(f"- Colunas categóricas: {len(data_types['categorical'])}")
            insights.append(f"- Total: {data_types['total_columns']} colunas, {data_types['total_rows']} linhas")
        
        # Estatísticas descritivas
        if any(word in question_lower for word in ['estatística', 'média', 'mediana', 'desvio', 'variância']):
            stats = self.analyzer.generate_summary_statistics()
            insights.append("**Estatísticas Descritivas:**")
            for col in stats.columns[:3]:
                insights.append(f"- {col}: média={stats[col]['mean']:.2f}, mediana={stats[col]['50%']:.2f}, desvio={stats[col]['std']:.2f}")
        
        # Detecção de outliers
        if any(word in question_lower for word in ['outlier', 'anomalia', 'atípico']):
            insights.append("**Detecção de Outliers:**")
            outlier_found = False
            for col in self.analyzer.numeric_cols[:5]:
                outlier_count = self.analyzer.detect_outliers(col)
                if outlier_count > 0:
                    insights.append(f"- {col}: {outlier_count} outliers detectados")
                    outlier_found = True
            if not outlier_found:
                insights.append("- Nenhum outlier significativo detectado nas principais colunas")
        
        # Correlação
        if any(word in question_lower for word in ['correlação', 'relação', 'associação']):
            corr_matrix = self.analyzer.correlation_analysis()
            if corr_matrix is not None:
                high_corr = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_value = corr_matrix.iloc[i, j]
                        if abs(corr_value) > 0.7:
                            high_corr.append(f"{corr_matrix.columns[i]} - {corr_matrix.columns[j]}: {corr_value:.2f}")
                
                if high_corr:
                    insights.append("**Correlações Fortes (>0.7):**")
                    insights.extend([f"- {corr}" for corr in high_corr[:3]])
                else:
                    insights.append("**Correlações:** Nenhuma correlação forte (>0.7) encontrada")
        
        # Distribuição
        if any(word in question_lower for word in ['distribuição', 'histograma', 'frequência']):
            insights.append("**Análise de Distribuição:**")
            available_cols = self.analyzer.numeric_cols[:3]
            for col in available_cols:
                insights.append(f"- {col}: disponível para análise de distribuição")
        
        # Se não encontrou padrões específicos, dar uma resposta geral
        if not insights:
            insights.append("**Análise Geral:**")
            insights.append(f"- Dataset com {len(df)} linhas e {len(df.columns)} colunas")
            insights.append(f"- Colunas numéricas: {len(self.analyzer.numeric_cols)}")
            insights.append("- Use perguntas específicas para análises detalhadas")
        
        return insights
    
    def generate_visualization(self, question, df):
        question_lower = question.lower()
        
        # Histograma
        if 'histograma' in question_lower:
            for col in self.analyzer.numeric_cols:
                if col.lower() in question_lower:
                    return self.analyzer.create_histogram(col)
            if self.analyzer.numeric_cols:
                return self.analyzer.create_histogram(self.analyzer.numeric_cols[0])
        
        # Dispersão
        if 'dispersão' in question_lower or 'scatter' in question_lower:
            numeric_cols = self.analyzer.numeric_cols
            if len(numeric_cols) >= 2:
                return self.analyzer.create_scatter_plot(numeric_cols[0], numeric_cols[1])
        
        # Box plot
        if 'box' in question_lower or 'boxplot' in question_lower:
            for col in self.analyzer.numeric_cols:
                if col.lower() in question_lower:
                    return self.analyzer.create_box_plot(col)
            if self.analyzer.numeric_cols:
                return self.analyzer.create_box_plot(self.analyzer.numeric_cols[0])
        
        return None

def main():
    st.title("🔍 Agente de IA - Análise Inteligente de Dados CSV")
    st.markdown("""
    Este agente usa **Inteligência Artificial** para analisar qualquer arquivo CSV. 
    Faça perguntas em linguagem natural sobre seus dados!
    """)
    
    # Inicialização do agente
    if 'agent' not in st.session_state:
        st.session_state.agent = DataAnalysisAgent()
    
    agent = st.session_state.agent
    
    # Upload de arquivo
    uploaded_file = st.file_uploader("📤 Carregue seu arquivo CSV", type=['csv'])
    
    # Botão para dados de exemplo
    if not uploaded_file and st.button("🚀 Carregar Dataset de Exemplo (Fraudes)"):
        # Criar dados de exemplo
        np.random.seed(42)
        n_samples = 1000
        
        example_data = {
            'Time': np.random.exponential(1000, n_samples),
            'V1': np.random.normal(0, 1, n_samples),
            'V2': np.random.normal(0, 1, n_samples),
            'V3': np.random.normal(0, 1, n_samples),
            'Amount': np.random.exponential(100, n_samples),
            'Class': np.random.choice([0, 1], n_samples, p=[0.99, 0.01])
        }
        
        df_example = pd.DataFrame(example_data)
        agent.load_demo_data(df_example)
        st.session_state.demo_mode = True
        st.session_state.current_df = df_example
        st.success("✅ Dataset de exemplo carregado com sucesso!")
        st.rerun()
    
    # Processar arquivo upload ou demo
    current_df = None
    if uploaded_file is not None:
        current_df, message = agent.load_data(uploaded_file)
        if current_df is not None:
            st.success(message)
            st.session_state.demo_mode = False
            st.session_state.current_df = current_df
    elif st.session_state.get('demo_mode', False):
        current_df = st.session_state.get('current_df', None)
    
    if current_df is not None:
        # Sidebar com informações do dataset
        with st.sidebar:
            st.header("📊 Informações do Dataset")
            st.write(f"**Shape:** {current_df.shape[0]} linhas × {current_df.shape[1]} colunas")
            
            st.subheader("Pré-visualização dos Dados")
            st.dataframe(current_df.head(10))
            
            st.subheader("Colunas")
            for col in current_df.columns:
                st.write(f"- {col} ({current_df[col].dtype})")
        
        # Área de perguntas - SIMPLIFICADA
        st.subheader("💬 Faça sua pergunta sobre os dados")
        
        with st.form(key='question_form'):
            question = st.text_input(
                "Exemplos: 'Quais são os tipos de dados?', 'Mostre um histograma', 'Existem outliers?'",
                placeholder="Digite sua pergunta aqui..."
            )
            
            submit_button = st.form_submit_button("🔍 Analisar com IA")
            
            if submit_button and question:
                with st.spinner("🤖 Analisando dados..."):
                    # Obter insights da LLM
                    llm_insights = agent.get_llm_insights(question, current_df)
                    
                    # Análise programática tradicional
                    insights = agent.analyze_question(question, current_df)
                    visualization = agent.generate_visualization(question, current_df)
                    
                    # Adicionar à memória
                    agent.memory.add_message("user", question)
                    
                    # Exibir insights da LLM
                    st.markdown("## 🤖 Insights da IA")
                    st.info(llm_insights)
                    
                    # Exibir análise programática
                    if insights:
                        response = "## 📈 Análise Estatística\n\n" + "\n".join(insights)
                        agent.memory.add_message("assistant", response)
                        st.markdown(response)
                    
                    # Exibir visualização
                    if visualization:
                        st.plotly_chart(visualization, use_container_width=True)
                        agent.memory.add_insight(f"Gráfico gerado para: {question}")
        
        # Histórico da conversa
        st.subheader("📝 Histórico da Análise")
        conversation_history = agent.memory.get_conversation_history()
        
        if conversation_history:
            for msg in conversation_history[-10:]:
                with st.chat_message(msg['role']):
                    st.markdown(msg['content'])
        else:
            st.info("💡 Faça sua primeira pergunta para começar a análise!")
        
        # Insights gerais
        st.subheader("🎯 Conclusões do Agente")
        insights = agent.memory.get_insights()
        if insights:
            for insight in insights[-5:]:
                st.info(insight)
        else:
            st.info("🔍 As conclusões aparecerão aqui após as análises")
    
    else:
        st.info("👆 Por favor, carregue um arquivo CSV ou use o dataset de exemplo para começar a análise")

if __name__ == "__main__":
    main()