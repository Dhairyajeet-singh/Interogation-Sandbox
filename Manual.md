HOW TO PLAY
===========

Someone is dead. Three or more people were there. One of them did it, the
others are hiding smaller things. You are the detective. You have a
limited number of questions before you must accuse.


THE SCREEN
----------

Top bar        the case name, your difficulty, turns left, score, and how
               many lies you have caught out of how many were planted.

               📁 case file  - everything you are allowed to know, and
                               the evidence board that grows as you play
               new case      - pick another from the library, or have
                               DeepSeek write a fresh one
               ?             - this manual

Suspect cards  click a card to question that person. The thin bar is
               their composure: green is calm, red is cornered. A
               cornered suspect talks differently - they ramble, or
               clam up. "dossier" opens everything you have on them.

The paper      the last exchange. The question, the answer, and what it
               got you: a severity band, facts added to the board, a
               contradiction caught, or "nothing new".

The machine    the right-hand panel. Not part of the game - it shows
               what the system is doing underneath. See "What is
               actually happening" below. Hide it with the toggle.


A TURN
------

1. Pick a suspect.

2. Press "question <name>". The system prepares three lines of
   questioning. You see the questions and, on easy and normal, what
   KIND each one is. You do NOT see how the suspect will answer or how
   useful it will be - that is the game.

      press claim         push on something they already said
      confront evidence   put a fact in front of them they did not
                          know you had
      revisit gap         make them tell it again, in order
      open topic          something they have not been asked about

   On hard, the kinds are hidden too.

3. Pick one and press "ask it". The answer appears on the paper, along
   with a band - HIGH, MEDIUM, LOW - saying how much that answer was
   worth. Facts the suspect let slip go onto the evidence board.

   Or type your own question in the box instead. Free questions are
   not scored, but they can still catch a lie.

Each of these costs one turn. Every turn costs 10 points, and a turn
that produced nothing new costs 15 more. Do not waste them.


CATCHING LIES
-------------

A contradiction is when a suspect says something that conflicts with
the physical record - the timeline of who was where - or with something
they said earlier. When it happens you will see "caught:" on the paper,
the suspect's composure drops, and you earn 100 points.

The lies are planted. Each suspect has at least one, and each can be
exposed by evidence they do not know you have. Confronting people with
things they think are secret is how you find them.


QUOTING
-------

Once someone has spoken, you can quote them at another suspect:
"quote Vance at Rourke". Rourke hears what Vance said and has to react.

This does two things. It puts pressure on. And it teaches Rourke
something - if Vance mentioned a time or a place Rourke did not know
about, Rourke now knows it, and it shows as "now knows" in his dossier.

That knowledge is real. Rewind past the quote and he forgets it again.


THE LAB
-------

The forensic tools run in a separate service and reason over the case
record, not over anyone's memory. Suspects cannot influence them.

      check alibi        does the record put this person at the scene
                         during the window?
      who was at         who does the record place at the scene?
      movements of       everywhere the record places this person
      verify statement   pull every checkable claim out of their last
                         answer and test each one
      lookup evidence    read the file on a piece of evidence

Anything they reveal goes on the evidence board. A verdict of
"contradicted" is pinned to that suspect's dossier and worth 50 points.
Each use costs a turn.


REWINDING
---------

Open a suspect's dossier and press "rewind to before this" on any
exchange. Everything from that point on is erased - for the suspect and
for you. They forget the questions, the board loses what those turns
added, and their composure goes back to what it was.

This is not a prompt trick. The suspect's memory is physically cut back.

Easy: rewinds are free and the turns come back.
Normal: the turns stay spent.
Hard: rewinding costs two more.


ACCUSING
--------

The buttons at the bottom of the room. Accusing ends the case.

The judge reads the whole transcript against the truth and decides not
just whether you named the right person, but whether you EARNED it:

      CASE CLOSED            right person, and the transcript shows
                             you established it            +500
      RIGHT NAME, THIN CASE  right person, but you never actually
                             caught them - a lucky guess    +150
      WRONG                                                 -200

The judge is deepseek-reasoner reading your transcript. With no API
key it falls back to a mechanical check and says so. It takes a minute
or two either way.


DIFFICULTY
----------

              turns   question kinds   rewind
      easy     15     shown            free, turns refunded
      normal   10     shown            turns stay spent
      hard      7     hidden           costs two more turns


SCORING
-------

      +100   each contradiction caught
       +50   each contradiction the lab confirms
       +25   each new fact on the evidence board
       -10   each turn
       -15   extra for a turn that revealed nothing
      +500 / +150 / -200   the accusation, as above


NEW CASES
---------

"new case" in the top bar. Pick one from the library, choose a
difficulty, press play.

Or set how many suspects you want and press "generate with DeepSeek".
deepseek-reasoner writes a case; a validator checks that it holds
together - the culprit had opportunity, nobody is in two places at
once, every planted lie can actually be discovered - and sends it back
for repair until it does. One to three minutes. It is then saved to the
library.

Without an API key you get a deterministic offline case instead, and
the case file is stamped to say so.


WHAT IS ACTUALLY HAPPENING
--------------------------

This is the part the game exists to show.

Every suspect is the same language model. When a model reads text it
builds an internal memory called a KV cache - one entry per word it has
read. Building that memory is the expensive part.

Normally each suspect would read the whole case file separately, and
re-read the entire conversation on every turn. Here the case file is
read ONCE. That memory is copied to every suspect, and each copy has
that suspect's private secret added on top. The machine panel shows
"saved on shared" - that is the reading work that never had to happen.

Rewinding is cutting that memory back. The words are gone.

"Question <name>" copies the suspect's memory three times, asks a
different question in each copy, scores the three answers, and shows
you the questions. Whichever you pick, its copy becomes the suspect's
real memory; the other two are thrown away. The branch tree draws
this. It is affordable only because copying is nearly free - without
the cache, three trial questions would mean reading the case file
three more times.

Snapshots pile up and the graphics card is small, so a store decides
which stay on the GPU, which get squeezed to 8-bit, which move to
system memory, and which are dropped. The snapshot store bar and the
cache log show it happening.

None of that is required to play. All of it is why the game works.