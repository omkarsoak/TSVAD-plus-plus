```tikz
\begin{document}
  \begin{tikzpicture}[font=\sffamily\bfseries, >=latex]
    \useasboundingbox (-3.4,-12.7) rectangle (7.9,0.9);
    \tikzset{box/.style={draw, rounded corners, minimum width=5.6cm, minimum height=0.9cm, text width=5.3cm, align=center, font=\sffamily\bfseries\scriptsize}}

    \node[box, fill=cyan!15] (b1) at (0,0) {DIHARD III audio\\and ground truth speaker turns};
    \node[box, fill=cyan!15] (b2) at (0,-1.7) {Speaker turns used for profiles:\\ground truth for dev,\\VBx clustering for eval};
    \node[box, fill=cyan!15] (b3) at (0,-3.4) {Cut single speaker audio\\ECAPA TDNN gives a 192 number\\profile for each speaker};
    \node[box, fill=cyan!15] (b4) at (0,-5.1) {Split dev recordings by domain:\\80 percent train, 20 percent validation};
    \node[box, fill=teal!20] (b5) at (0,-6.8) {Train on the train split\\validate every epoch\\keep the best model};
    \node[box, fill=teal!20] (b6) at (0,-8.5) {Average the 5 best checkpoints};
    \node[box, fill=teal!20] (b7) at (0,-10.2) {Tune threshold, median filter and\\gap settings on validation only};
    \node[box, fill=brown!20] (b8) at (0,-11.9) {Score the eval set once\\DER overall and per domain};

    \node[draw, rounded corners, fill=gray!20, minimum width=3.2cm, minimum height=0.9cm, text width=2.9cm, align=center, font=\sffamily\bfseries\scriptsize] (s1) at (5.8,-6.8) {Optional:\\pretrain on simulated\\mixtures first};

    \draw[->] (b1) -- (b2);
    \draw[->] (b2) -- (b3);
    \draw[->] (b3) -- (b4);
    \draw[->] (b4) -- (b5);
    \draw[->] (b5) -- (b6);
    \draw[->] (b6) -- (b7);
    \draw[->] (b7) -- (b8);
    \draw[->, dashed] (s1.west) -- (b5.east);
  \end{tikzpicture}
\end{document}
```
