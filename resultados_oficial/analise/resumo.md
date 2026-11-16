# Resultados (metade de teste)

Projetos por versão: {'A': 15, 'B': 15, 'C': 15}

## Qualidade (média ± desvio)

| versao   | qualidade_experimental   | qualidade_relatorio   | utilidade   | nota_humana   | nota_llm    | mle_score   |
|:---------|:-------------------------|:----------------------|:------------|:--------------|:------------|:------------|
| A        | 1.07 ± 1.28              | 1.2 ± 1.47            | 0.8 ± 0.94  | 1.02 ± 1.2    | 3.29 ± 1.38 | 0.05 ± 0.14 |
| B        | 0.6 ± 0.99               | 0.8 ± 1.21            | 0.53 ± 0.92 | 0.64 ± 1.0    | 3.4 ± 1.14  | 0.11 ± 0.21 |
| C        | 0.67 ± 1.23              | 0.67 ± 1.18           | 0.47 ± 0.83 | 0.6 ± 1.05    | 3.0 ± 0.82  | 0.02 ± 0.06 |

## Falhas, custo e tempo por projeto (média ± desvio)

| versao   | sucesso_etapas   | falhas       | falhas_repetidas   | tokens               | tempo_min     |
|:---------|:-----------------|:-------------|:-------------------|:---------------------|:--------------|
| A        | 0.79 ± 0.2       | 9.07 ± 6.31  | 8.47 ± 5.5         | 189514.2 ± 55732.86  | 39.61 ± 26.21 |
| B        | 0.63 ± 0.29      | 10.27 ± 7.19 | 8.0 ± 5.14         | 212565.8 ± 60661.37  | 57.28 ± 40.39 |
| C        | 0.58 ± 0.25      | 9.07 ± 6.35  | 8.33 ± 6.13        | 204925.73 ± 49481.84 | 61.55 ± 46.95 |

## Falhas por projeto, por etapa

| fase                |    A |    B |    C |
|:--------------------|-----:|-----:|-----:|
| data_preparation    | 6.2  | 6.27 | 4.87 |
| literature_review   | 1    | 1.6  | 1.8  |
| report_writing      | 0.53 | 0.67 | 0.73 |
| running_experiments | 1.33 | 1.73 | 1.67 |

## Falhas por categoria (total)

| categoria           |   A |   B |   C |
|:--------------------|----:|----:|----:|
| acuracia_zero       |   1 |   1 |   2 |
| busca_em_loop       |   8 |  15 |  14 |
| dependencia_ausente |  13 |   8 |   0 |
| erro_de_execucao    |  23 |  17 |  21 |
| erro_de_sintaxe     |  56 |  44 |  42 |
| fase_falhou         |  10 |  18 |  21 |
| formato_invalido    |   5 |   1 |   3 |
| outro_erro          |   3 |   6 |   2 |
| sem_relatorio       |   8 |  10 |  11 |
| timeout             |   9 |  34 |  20 |

## Testes estatísticos (nota_humana, 15 pares)

- Friedman A×B×C: estatística=1.69, p=0.430
- Wilcoxon B vs A: diferença média=-0.38, p=0.259
- Wilcoxon C vs A: diferença média=-0.42, p=0.174
- Wilcoxon C vs B: diferença média=-0.04, p=0.943
- Revisor LLM vs. humanos (Spearman): rho=-0.15, p=0.589; média LLM=3.25/10 vs. humana=2.12/5

Com 5 temas por seed o poder estatístico é baixo: trate os p-valores como indicativos.
