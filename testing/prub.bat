@echo off
echo %time%
echo %~dp0
cd C:\Program Files\Blender Foundation\Blender 4.3\
for /L %%A in (0,1,290) do (
  echo %%A
  blender -w "%~dp0\GUI.blend" --python "%~dp0\automat.py3"
  timeout 2 > NUL
)
echo %time%
