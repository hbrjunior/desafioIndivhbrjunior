# app.py - VERSÃO CORRIGIDA
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
import io
import base64
from datetime import datetime
import warnings
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
        return self.df[self.numeric_cols].corr()
    
    def create_histogram(self, column):
        fig = px.histogram(self.df, x=column, title=f'Distribuição de {column}')
        return fig
    
    def create_scatter_plot(self, x_col, y_col):
        fig = px.scatter(self.df, x=x_col, y=y_col, title=f'{x_col} vs {y_col}')
        return fig
    
    def create_box_plot(self, column):
        fig = px.box(self.df, y=column, title=f'Box Plot - {column}')
        return fig

# Agente principal
class DataAnalysisAgent:
    def __init__(self):
        self.memory = ConversationMemory()
        self.analyzer = None
    
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
            for col in stats.columns[:3]:  # Mostra apenas as 3 primeiras colunas para não sobrecarregar
                insights.append(f"- {col}: média={stats[col]['mean']:.2f}, mediana={stats[col]['50%']:.2f}")
        
        # Detecção de outliers
        if any(word in question_lower for word in ['outlier', 'anomalia', 'atípico']):
            insights.append("**Detecção de Outliers:**")
            for col in self.analyzer.numeric_cols[:3]:
                outlier_count = self.analyzer.detect_outliers(col)
                if outlier_count > 0:
                    insights.append(f"- {col}: {outlier_count} outliers detectados")
        
        # Correlação
        if any(word in question_lower for word in ['correlação', 'relação', 'associação']):
            corr_matrix = self.analyzer.correlation_analysis()
            high_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.7:
                        high_corr.append(f"{corr_matrix.columns[i]} - {corr_matrix.columns[j]}: {corr_matrix.iloc[i, j]:.2f}")
            
            if high_corr:
                insights.append("**Correlações Fortes (>0.7):**")
                insights.extend([f"- {corr}" for corr in high_corr[:3]])
        
        # Distribuição
        if any(word in question_lower for word in ['distribuição', 'histograma', 'frequência']):
            insights.append("**Análise de Distribuição:**")
            for col in self.analyzer.numeric_cols[:2]:
                insights.append(f"- {col}: disponível para análise de distribuição")
        
        return insights
    
    def generate_visualization(self, question, df):
        question_lower = question.lower()
        
        if 'histograma' in question_lower:
            for col in self.analyzer.numeric_cols:
                if col.lower() in question_lower:
                    return self.analyzer.create_histogram(col)
        
        if 'dispersão' in question_lower or 'scatter' in question_lower:
            numeric_cols = self.analyzer.numeric_cols
            if len(numeric_cols) >= 2:
                return self.analyzer.create_scatter_plot(numeric_cols[0], numeric_cols[1])
        
        if 'box' in question_lower or 'boxplot' in question_lower:
            for col in self.analyzer.numeric_cols:
                if col.lower() in question_lower:
                    return self.analyzer.create_box_plot(col)
        
        # Gráfico padrão - histograma da primeira coluna numérica
        if self.analyzer.numeric_cols:
            return self.analyzer.create_histogram(self.analyzer.numeric_cols[0])
        
        return None

# Interface principal
def main():
    st.title("🔍 Agente de IA para Análise de Dados CSV")
    st.markdown("""
    Este agente permite analisar qualquer arquivo CSV, gerar visualizações e insights automáticos.
    Faça perguntas sobre seus dados e obtenha análises detalhadas!
    """)
    
    # Inicialização do agente
    if 'agent' not in st.session_state:
        st.session_state.agent = DataAnalysisAgent()
    
    agent = st.session_state.agent
    
    # Upload de arquivo
    uploaded_file = st.file_uploader("📤 Carregue seu arquivo CSV", type=['csv'])
    
    if uploaded_file is not None:
        df, message = agent.load_data(uploaded_file)
        
        if df is not None:
            st.success(message)
            
            # Sidebar com informações do dataset
            with st.sidebar:
                st.header("📊 Informações do Dataset")
                st.write(f"**Shape:** {df.shape[0]} linhas × {df.shape[1]} colunas")
                
                st.subheader("Pré-visualização dos Dados")
                st.dataframe(df.head(10))
                
                st.subheader("Colunas")
                for col in df.columns:
                    st.write(f"- {col} ({df[col].dtype})")
            
            # Área de perguntas e respostas
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("💬 Faça sua pergunta")
                
                # Usar um key único para evitar conflitos
                question = st.text_input(
                    "Exemplos: 'Quais são os tipos de dados?', 'Mostre um histograma da coluna X', 'Existem outliers?'",
                    key="user_question_input"
                )
                
                # Verificar se há pergunta selecionada das sugestões
                if 'selected_question' in st.session_state and st.session_state.selected_question:
                    question = st.session_state.selected_question
                    st.session_state.selected_question = ""  # Limpar após usar
                
                if st.button("Analisar", type="primary") and question:
                    with st.spinner("Analisando dados..."):
                        # Processar pergunta
                        insights = agent.analyze_question(question, df)
                        visualization = agent.generate_visualization(question, df)
                        
                        # Adicionar à memória
                        agent.memory.add_message("user", question)
                        
                        # Exibir resposta
                        if insights:
                            response = "## 📈 Análise dos Dados\n\n" + "\n".join(insights)
                            agent.memory.add_message("assistant", response)
                            st.markdown(response)
                        
                        # Exibir visualização
                        if visualization:
                            st.plotly_chart(visualization, use_container_width=True)
                            
                            # Adicionar insight sobre a visualização
                            agent.memory.add_insight(f"Gráfico gerado para: {question}")
            
            with col2:
                st.subheader("💡 Perguntas Sugeridas")
                suggested_questions = [
                    "Quais são os tipos de dados?",
                    "Mostre estatísticas descritivas",
                    "Existem outliers nos dados?",
                    "Mostre um histograma",
                    "Quais as correlações entre variáveis?",
                    "Qual a distribuição dos dados?"
                ]
                
                for q in suggested_questions:
                    if st.button(q, key=f"suggest_{hash(q)}"):
                        # Usar approach seguro para atualizar a pergunta
                        st.session_state.selected_question = q
                        st.rerun()
            
            # Histórico da conversa
            st.subheader("📝 Histórico da Análise")
            conversation_history = agent.memory.get_conversation_history()
            
            for msg in conversation_history[-10:]:  # Mostrar últimas 10 mensagens
                with st.chat_message(msg['role']):
                    st.markdown(msg['content'])
            
            # Insights gerais
            st.subheader("🎯 Conclusões do Agente")
            insights = agent.memory.get_insights()
            if insights:
                for insight in insights[-5:]:  # Mostrar últimos 5 insights
                    st.info(insight)
            else:
                st.info("Faça perguntas para gerar insights sobre os dados!")
    
    else:
        st.info("👆 Por favor, carregue um arquivo CSV para começar a análise")
        
        # Exemplo com dados de demonstração
        if st.button("Carregar Dataset de Exemplo (Fraudes)"):
            # Criar dados de exemplo baseados no dataset de fraudes
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
            csv_buffer = StringIO()
            df_example.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)
            
            # Simular upload
            st.session_state.uploaded_file = csv_buffer
            st.rerun()

if __name__ == "__main__":
    main()