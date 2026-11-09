```tikz
\begin{document}
  \begin{tikzpicture}[font=\sffamily\bfseries, >=latex]
    \useasboundingbox (-5.5,-15.4) rectangle (5.5,0.9);
    \tikzset{box/.style={draw, rounded corners, minimum height=0.9cm, align=center, font=\sffamily\bfseries\scriptsize}}

    \node[box, fill=cyan!15, minimum width=4.8cm, text width=4.5cm] (a1) at (-2.7,0) {Audio chunk\\4 s at 16 kHz};
    \node[box, fill=cyan!15, minimum width=4.8cm, text width=4.5cm] (p1) at (2.7,0) {Speaker profiles\\8 x 192 from ECAPA TDNN};

    \node[box, fill=teal!20, minimum width=4.8cm, text width=4.5cm] (w1) at (-2.7,-1.6) {WavLM, first 6 layers\\768 numbers per 20 ms};
    \node[box, fill=teal!20, minimum width=4.8cm, text width=4.5cm] (c1) at (-2.7,-3.2) {Conv1d stride 2\\T x 192, T = 100 frames of 40 ms};
    \node[box, fill=teal!20, minimum width=4.8cm, text width=4.5cm] (r1) at (2.7,-1.6) {Repeat each profile\\over all T frames};

    \node[box, fill=teal!20, minimum width=6.8cm, text width=6.5cm] (cat) at (0,-4.9) {Concatenate speech and profile\\one stream per speaker, 384 per frame};
    \node[box, fill=blue!15, minimum width=6.8cm, text width=6.5cm] (t1) at (0,-6.5) {Transformer 1, 3 layers\\applied to each speaker separately\\shared weights};
    \node[box, fill=teal!20, minimum width=6.8cm, text width=6.5cm] (st) at (0,-8.2) {Stack the 8 speaker streams\\Conv1d down to 384 per frame};
    \node[box, fill=blue!15, minimum width=6.8cm, text width=6.5cm] (t2) at (0,-9.8) {Transformer 2, 3 layers\\all speakers together\\handles overlap};
    \node[box, fill=brown!20, minimum width=6.8cm, text width=6.5cm] (hd) at (0,-11.4) {Linear 384 to 8, sigmoid\\one output per speaker per frame};
    \node[box, fill=gray!20, minimum width=6.8cm, text width=6.5cm] (out) at (0,-12.9) {Probability that speaker k talks\\at frame t, size 8 x T};
    \node[box, fill=gray!20, minimum width=6.8cm, text width=6.5cm] (pp) at (0,-14.4) {Median filter, threshold, fill gaps\\RTTM file};

    \draw[->] (a1) -- (w1);
    \draw[->] (w1) -- (c1);
    \draw[->] (p1) -- (r1);
    \draw[->] (c1.south) -- (cat.north -| c1);
    \draw[->] (r1.south) -- (cat.north -| r1);
    \draw[->] (cat) -- (t1);
    \draw[->] (t1) -- (st);
    \draw[->] (st) -- (t2);
    \draw[->] (t2) -- (hd);
    \draw[->] (hd) -- (out);
    \draw[->] (out) -- (pp);
  \end{tikzpicture}
\end{document}
```
