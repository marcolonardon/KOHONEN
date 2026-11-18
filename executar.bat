@echo off
chcp 65001 >nul
rem Executa o SOM de Kohonen com um duplo clique.
rem Na primeira vez cria o ambiente virtual e instala as dependências;
rem nas seguintes apenas roda o script e abre a pasta de resultados.
rem Opções do script podem ser repassadas: executar.bat --linhas 20 --colunas 20
setlocal
cd /d "%~dp0"

set "PY=.venv\Scripts\python.exe"

if not exist "%PY%" (
    echo Preparando o ambiente Python pela primeira vez...
    py -3 -m venv .venv 2>nul || python -m venv .venv
)
if not exist "%PY%" (
    echo.
    echo ERRO: não foi possível criar o ambiente virtual.
    echo Instale o Python 3 em https://www.python.org/downloads/ e tente de novo.
    pause
    exit /b 1
)

"%PY%" -c "import numpy, matplotlib" 2>nul
if errorlevel 1 (
    echo Instalando as dependências, aguarde...
    "%PY%" -m pip install --quiet --disable-pip-version-check -r requirements.txt
    if errorlevel 1 (
        echo.
        echo ERRO: não foi possível instalar as dependências. Verifique a conexão com a internet.
        pause
        exit /b 1
    )
)

"%PY%" kohonen_som.py %*
if errorlevel 1 (
    echo.
    echo ERRO: a execução falhou. Veja a mensagem acima.
    pause
    exit /b 1
)

echo.
echo Abrindo a pasta de resultados...
start "" "%~dp0resultados"
pause
