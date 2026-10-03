# Esse projeto precisa seguir as seguintes diretrizes:

## Padrão de commits
Seguir commitizen

## CICD
Esteira precisa seguir as seguintes regras:
- Branch main (prod) só aceita merge com a branch "dev"
- Cada vez que um commit no arquivo main.py é dado, dispara o gatilho que atualiza todo o processo da branch main, e se for um commit na branch dev, atualiza todo o processo na branch dev.