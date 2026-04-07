@echo off
setlocal enabledelayedexpansion

:: === CONFIGURAÇÕES (altere se necessário) ===
set "CKPT=C:\Users\guico\Documents\code\F5TTS-testes\prosodic\model_last.pt"
set "VOCAB=C:\Users\guico\Documents\code\F5TTS-testes\F5-TTS\f5tts_v1\vocab.txt"
set "REF=C:\Users\guico\Downloads\sp_did_234_amostra_10s_22khz.wav"
set "OUTDIR=C:\Users\guico\Documents\code\F5TTS-testes\prosodic\audios\SP_DID_234"

:: === FRASES ===
set "PH1=eu quase não vou ao cinema teatro..."
set "PH2=ah às vezes eu vou..."
set "PH3=eu tenho ido a teatro."
set "PH4=deve ser como na televisão"
set "PH5=então no teatro eu acho que é bem mais difícil..."
set "PH6=a televisão é horroroso quando eles estão fazendo programa."
set "PH7=eu sei que não há preparação toda."
set "PH8=porque o grupo que trabalha em hair é enorme né"
set "PH9=tenho impressão que ali levou tanto tempo de ensaio"
set "PH10=me chocou tremendamente"
set "PH11=eu saber que o filme é bom"
set "PH12=eu gostei bastante"
set "PH13=eu me lembro de vários filmes não lembro os nomes"
set "PH14=por isso é que eu deixo de ir ao cinema"
set "PH15=hoje tá tudo meio louco né"
set "PH16=assisti em araraquara."
set "PH17=eu num lembro o nome do filme..."
set "PH18=a molecada adorou."
set "PH19=eles adoraram o filme..."
set "PH20=porque eu saio cansada mesmo"
set "PH21=fico numa tensão nervosa"
set "PH22=nós saímos pra ir ao teatro."
set "PH23=não conseguimos entrar fomos assistir esse filme."
set "PH24=eu acho que influi bastante"
set "PH25=eu acho que teatro tá bem mais caro"
set "PH26=eu acho que o público pre prefere cinema ainda"
set "PH27=eu não entendi a pergunta"
set "PH28=eu acho que o cinema tá perdendo viu"
set "PH29=o que eu noto é isso"
set "PH30=principalmente nos fins de semana"

:: === LOOP DE GERAÇÃO ===
echo Iniciando geração dos audios para 30 frases...
echo Checkpoint: %CKPT%
echo Vocab:      %VOCAB%
echo Ref audio:  %REF%
echo Output dir: %OUTDIR%
echo.

for /L %%i in (1,1,30) do (
    :: obter a frase correspondente (usa CALL para expandir a variável dinâmica PH%%i)
    CALL SET "TEXT=%%PH%%i%%%"
    echo ==============================================
    echo Gerando: %%i.wav
    echo Texto: !TEXT!
    echo ----------------------------------------------
    :: chamada ao comando (ajuste --model ou --model_cfg se precisar)
    f5-tts_infer-cli --ckpt_file "!CKPT!" --vocab_file "!VOCAB!" --ref_audio "!REF!" --ref_text "" --gen_text "!TEXT!" --output_dir "!OUTDIR!" --output_file %%i.wav
    echo.
)

echo ==============================================
echo Concluído!
pause
