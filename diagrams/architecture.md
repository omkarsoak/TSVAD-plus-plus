```tikz
\begin{document}
  \begin{tikzpicture}[font=\sffamily\bfseries, >=latex]
    \useasboundingbox (-5.9,-0.9) rectangle (5.6,15.9);
    \tikzset{box/.style={draw, rounded corners, minimum height=0.9cm, align=center, font=\sffamily\bfseries\scriptsize}}
    \tikzset{sbox/.style={draw, rounded corners, minimum height=0.7cm, minimum width=1.2cm, align=center, font=\sffamily\bfseries\scriptsize}}

    % inputs and encoders
    \node[box, fill=gray!20, minimum width=3.6cm, text width=3.3cm] (raw) at (-3.8,0) {Raw speech\\4 s chunk, 16 kHz};
    \node[box, fill=red!15, minimum width=5.4cm, text width=5.1cm] (tg) at (2.4,0) {Target speech\\top 8 speakers by talk time\\short speakers keep a profile};
    \node[box, fill=teal!20, minimum width=3.6cm, text width=3.3cm] (wl) at (-3.8,2.1) {WavLM, first 6 layers\\Conv1d to T x 192};
    \node[box, fill=teal!20, minimum width=5.4cm, text width=5.1cm] (ec) at (2.4,2.1) {ECAPA TDNN speaker model\\192 numbers per speaker};

    % speaker profiles
    \node[sbox, fill=cyan!20] (s1) at (0.7,3.8) {SE 1};
    \node[sbox, fill=teal!20] (s2) at (2.1,3.8) {SE 2};
    \node at (3.1,3.8) {....};
    \node[sbox, fill=gray!25] (s8) at (4.1,3.8) {SE 8};

    % speaker detection
    \node[box, fill=blue!15, minimum width=5.4cm, text width=5.1cm] (td) at (2.4,5.6) {Speaker Detection Transformer\\3 layers, shared by all speakers};
    \draw[rounded corners] (-0.1,6.95) rectangle (4.9,7.85);
    \node[sbox, fill=cyan!20] (d1) at (0.7,7.4) {SD 1};
    \node[sbox, fill=teal!20] (d2) at (2.1,7.4) {SD 2};
    \node at (3.1,7.4) {....};
    \node[sbox, fill=gray!25] (d8) at (4.1,7.4) {SD 8};

    % combining and output head
    \node[box, fill=blue!15, minimum width=5.4cm, text width=5.1cm] (cm) at (2.4,9.3) {Combining Transformer\\3 layers, all speakers together\\Conv1d 8 x 384 down to 384};
    \node[box, fill=red!15, minimum width=5.4cm, text width=5.1cm] (hd) at (2.4,11.3) {Per frame head\\Linear 384 to 8, sigmoid};

    % outputs and post processing
    \node[sbox, fill=cyan!20] (o1) at (0.7,13.0) {Out 1};
    \node[sbox, fill=teal!20] (o2) at (2.1,13.0) {Out 2};
    \node at (3.1,13.0) {....};
    \node[sbox, fill=gray!25] (o8) at (4.1,13.0) {Out 8};
    \node[box, fill=red!15, minimum width=5.4cm, text width=5.1cm] (pp) at (2.4,14.9) {Post processing to RTTM\\median filter, threshold, gap fill\\settings tuned on validation};

    % what TS-VAD++ adds
    \node[box, fill=red!15, minimum width=3.6cm, text width=3.3cm] (lg) at (-3.8,11.3) {Red marks TS-VAD++ additions\\Per frame head\\Short speaker profiles\\Validation tuned settings\\Best 5 checkpoints averaged};

    % arrows
    \draw[->] (raw) -- (wl);
    \draw[->] (tg) -- (ec);
    \draw[->] (ec.north -| s1) -- (s1.south);
    \draw[->] (ec.north -| s2) -- (s2.south);
    \draw[->] (ec.north -| s8) -- (s8.south);
    \draw[->] (s1.north) -- (td.south -| s1);
    \draw[->] (s2.north) -- (td.south -| s2);
    \draw[->] (s8.north) -- (td.south -| s8);
    \draw[->] (wl.north) |- (td.west);
    \draw[->] (td.north -| d1) -- (d1.south);
    \draw[->] (td.north -| d2) -- (d2.south);
    \draw[->] (td.north -| d8) -- (d8.south);
    \draw[->] (d1.north) -- (cm.south -| d1);
    \draw[->] (d2.north) -- (cm.south -| d2);
    \draw[->] (d8.north) -- (cm.south -| d8);
    \draw[->] (cm) -- (hd);
    \draw[->] (hd.north) -- (o1.south);
    \draw[->] (hd.north) -- (o2.south);
    \draw[->] (hd.north) -- (o8.south);
    \draw[->] (o1.north) -- (pp.south -| o1);
    \draw[->] (o2.north) -- (pp.south -| o2);
    \draw[->] (o8.north) -- (pp.south -| o8);
  \end{tikzpicture}
\end{document}
```
