#  Análise de Dados do E-commerce Brasileiro - Olist

## Descrição do Projeto

Este projeto tem como objetivo explorar e analisar dados de vendas e logística do e-commerce brasileiro **Olist**. Utilizando dados reais de clientes, pedidos, produtos, avaliações e localização, o projeto busca identificar padrões de comportamento de consumidores, avaliar a eficiência da logística e sugerir estratégias para melhorar a performance da empresa.

Com base nas análises realizadas, são fornecidas recomendações práticas para otimizar a experiência do cliente e melhorar os processos logísticos, com foco na redução de atrasos na entrega, segmentação de clientes e estratégias de marketing mais eficazes.

### Principais Análises Realizadas:

1. **Relação entre Atrasos na Entrega e Avaliação dos Clientes**:
   - Identificação do impacto dos atrasos no tempo de entrega na avaliação dos consumidores, mostrando que a pontualidade é um fator crucial para a experiência de compra.

2. **Comportamento de Compra por Região**:
   - Análise da variação no volume de vendas e ticket médio por região, sugerindo a necessidade de estratégias de marketing segmentadas e adequadas às particularidades de cada localidade.

3. **Segmentação de Clientes (RFM)**:
   - Aplicação da técnica de Recência, Frequência e Valor Monetário (RFM) para identificar os clientes mais valiosos e aqueles com maior risco de churn, permitindo a criação de campanhas específicas para aumentar a retenção.

4. **Eficiência Logística e Performance das Regiões**:
   - Estudo sobre o desempenho da logística, com foco nos tempos de entrega por região, destacando áreas que demandam mais atenção para melhorar a eficiência.

## Tecnologias Utilizadas

- **Python**: Linguagem principal para análise de dados e visualização.
- **Pandas**: Manipulação e limpeza dos dados.
- **Plotly**: Geração de gráficos interativos e visualizações.
- **Streamlit**: Criação de um dashboard interativo para visualização e apresentação dos resultados.

## Estrutura do Projeto

O projeto é estruturado da seguinte forma:

1. **Carregamento e Tratamento de Dados**: 
   - Leitura e tratamento dos dados provenientes de diferentes datasets relacionados a clientes, produtos, pedidos e avaliações.
   
2. **Exploração dos Dados**: 
   - Análises exploratórias para entender o comportamento dos clientes, a eficiência logística e as avaliações de produtos.

3. **Visualizações Interativas**: 
   - Utilização de gráficos dinâmicos com a biblioteca **Plotly**, proporcionando uma experiência interativa para explorar os dados.
   
4. **Recomendações Práticas**: 
   - A partir das análises, são fornecidas recomendações para a empresa Olist, visando melhorar a performance de vendas e logística.

## Como Executar o Projeto

1. Clone o repositório:
   git clone https://github.com/seu_usuario/olist-analysis.git

2. Instale as dependências:
    pip install -r index/requirements.txt

3. Execute o script principal no Streamlit:
    streamlit run app.py

4. Acesse o dashboard gerado no navegador, onde as visualizações interativas estarão disponíveis.

## Conclusão

Este projeto oferece uma visão detalhada sobre o comportamento do cliente e a performance logística da Olist. Ele fornece uma base sólida para tomadas de decisão mais informadas e estratégias de melhoria contínua.

### Recomendações para a Olist:
- Melhorar a logística nas regiões com maiores atrasos, como Norte e Centro-Oeste, para reduzir o impacto no tempo de entrega e melhorar a satisfação dos clientes.
- Criar campanhas direcionadas às regiões com maior ticket médio, como Norte e Nordeste, oferecendo incentivos para aumentar a frequência de compra.
- Implementar estratégias específicas de retenção para os grupos de clientes "em risco", oferecendo promoções e vantagens exclusivas.

Apesar dos insights gerados, algumas limitações foram observadas, como a ausência de dados sobre transportadoras e a falta de informações mais granulares sobre o comportamento dos clientes. No futuro, seria interessante incluir dados de comportamento online e feedback dos consumidores para expandir ainda mais a análise e prever tendências futuras de vendas.

## Contribuições

Sinta-se à vontade para contribuir com este projeto! Para sugestões, correções ou melhorias, basta abrir uma issue ou enviar um pull request.

## Contato
- **LinkedIn**: [Adriano Mayco](https://www.linkedin.com/in/adriano-mayco-382256221/)
- **E-mail**: adri.mayco@protonmail.com

