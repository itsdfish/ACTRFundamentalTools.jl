;; (load "/home/dfish/.julia/dev/ACTRTutorial/Utilities/actr6/load-act-r-6.lisp")
;; (load "/home/dfish/.julia/dev/FundamentalToolsACTR/models/Semantic/semantic.lisp")

(clear-all)

(defvar *response*)
(defvar *response-time*)
(defvar *rt* 0.0)

(defun run-semantic-model (term1 term2)
  (let ((window (open-exp-window "Semantic Model"
                                 :visible nil
                                 :width 600
                                 :height 300))
        (x 25))

    (reset)
    (install-device window)

   (dolist (text (list term1  term2))
     (add-text-to-exp-window :text text :x x :y 150 :width 75)
     (incf x 75))

    (setf *response* nil)
    (setf *response-time* nil)

    (proc-display)

    (run 30)
    (if (string-equal *response* "j")
        (setf present "yes")
        (setf present "no"))

    (if (null *response*)
        nil
      present)))

(defmethod rpm-window-key-event-handler ((win rpm-window) key)
  (setf *response-time* (get-time t))

(setf *response* (string key)))

(defun generate-data (n-reps)
    (let ((stimuli (list)) (results (list)))
    (setf stimuli (list '("canary" "fish" "yes")))
    (dolist (stimulus stimuli)
        (push (run-n-times n-reps (nth 0 stimulus)
            (nth 1 stimulus) (nth 2 stimulus)) results)
    )
    results))

(defun generate-recovery-data (n-parms n-reps path)
  (sgp :seed (584001 98))
  (let ((responses (list)) (file-name) (all-parms (list (list "rt"))) (parms (list)))
    (dotimes (i n-parms)
      (setf responses (list))
      (setf *rt* (sample -1.0 1.0))
      (setf parms (list *rt*))
      (setf all-parms (append all-parms (list parms)))
      (setf responses (append responses (generate-data n-reps)))
      (setf file-name (concatenate 'string path "/data_set_" (write-to-string i) "_.csv"))
      (write-to-file responses file-name))
      (setf file-name (concatenate 'string path "/true_parms.csv"))
      (write-to-file all-parms file-name)))

(defun run-n-times (n object category answer)
   (let ((num-yes 0) (response))
   (dotimes (i n)
      (setf response (run-semantic-model object category))
      (if (equal response "yes")
        (incf num-yes)))
      (list object category answer n num-yes)))

(defun sample (lb ub)
  (let ((val))
    (setf val (+ (act-r-random (- ub lb)) lb))
  ))

(defun penalty-fct (chunk request)
  (let ((chunk-type) (slot-value) (val) (mp))
      (setf chunk-type (chunk-spec-chunk-type request))
      (setf mp (first (sgp :mp)))
      (cond ((eq chunk-type 'meaning)
              (setf val (compute-penalty chunk request (* mp 20))))
          ((neq chunk-type 'meaning)
              (setf val (compute-penalty chunk request mp))))

      val))

(defun compute-penalty (chunk request scale)
  (let ((penalty 0) (slot-value))
  (dolist (k (chunk-spec-slot-spec request))
      (setf slot-value (fast-chunk-slot-value-fct chunk (second k)))
      (cond ((eq slot-value nil)
          (incf penalty scale))
      ((neq slot-value nil)
          (when (not (chunk-slot-equal slot-value (third k)))
              (incf penalty scale)))))
      (* -1 penalty)))

(defun write-to-file (lst FileName)
(with-open-file (file FileName
                      :direction :output
                      :if-exists :supersede
                      :if-does-not-exist :create)
  (loop for row in lst
    do (loop for n in row
         do (princ n file)
         (princ "," file))
         (fresh-line file))))

(defun read-csv (filename delim-char)
  "Reads the csv to a nested list, where each sublist represents a line."
(with-open-file (input filename)
  (loop :for line := (read-line input nil) :while line
        :collect (read-from-string
                  (substitute #\SPACE delim-char
                              (format nil "(~a)~%" line))))))

(define-model semantic

(sgp-fct (list :esc t :rt *rt* :ans .2 :v t :act nil :blc 1.0 :mp 1.0))
(sgp :partial-matching-hook penalty-fct)

(chunk-type property object attribute value)
(chunk-type is-member object category judgment)
(chunk-type question object category judgment)
(chunk-type meaning word)

(add-dm
 (shark isa meaning word "shark") (dangerous isa meaning word "dangerous")
 (locomotion isa meaning word "locomotion") (swimming isa meaning word "swimming")
 (fish isa meaning word "fish") (salmon isa meaning word "salmon")
 (edible isa meaning word "edible") (breathe isa meaning word "breathe")
 (gills isa meaning word "gills") (animal isa meaning word "animal")
 (moves isa meaning word "moves") (skin isa meaning word "skin")
 (color isa meaning word "color") (sings isa meaning word "sings")
 (ostrich isa meaning word "ostrich") (flies isa meaning word "flies")
 (height isa meaning word "height") (tall isa meaning word "tall")
 (wings isa meaning word "wings") (flying isa meaning word "flyinig")
 (true isa meaning word "true") (false isa meaning word "false")
 (canary isa meaning word "canary") (bird isa meaning word "bird")
 (p1 ISA property object shark attribute dangerous value true)
 (p2 ISA property object shark attribute locomotion value swimming)
 (p3 ISA property object shark attribute category value fish)
 (p4 ISA property object salmon attribute edible value true)
 (p5 ISA property object salmon attribute locomotion value swimming)
 (p6 ISA property object salmon attribute category value fish)
 (p7 ISA property object fish attribute breathe value gills)
 (p8 ISA property object fish attribute locomotion value swimming)
 (p9 ISA property object fish attribute category value animal)
 (p10 ISA property object animal attribute moves value true)
 (p11 ISA property object animal attribute skin value true)
 (p12 ISA property object canary attribute color value yellow)
 (p13 ISA property object canary attribute sings value true)
 (p14 ISA property object canary attribute category value bird)
 (p15 ISA property object ostrich attribute flies value false)
 (p16 ISA property object ostrich attribute height value tall)
 (p17 ISA property object ostrich attribute category value bird)
 (p18 ISA property object bird attribute wings value true)
 (p19 ISA property object bird attribute locomotion value flying)
 (p20 ISA property object bird attribute category value animal)
 (goal1 ISA is-member))

 (P find-object
     =goal>
        ISA         is-member
    ?visual-location>
        buffer      unrequested
    ?imaginal>
        state       free
    ==>
    +imaginal>
        ISA         question
    +visual-location>
        ISA         visual-location
        screen-x    lowest
 )

 (P attend-visual-location
    =visual-location>
        ISA         visual-location
    ?visual-location>
        buffer      requested
    ?visual>
        state       free
    ==>
    +visual>
        ISA         move-attention
        screen-pos  =visual-location
 )

 (P retrieve-meaning
     =visual>
        ISA         text
        value       =word
    ==>
     +retrieval>
        ISA        meaning
        word       =word
 )

 (P encode-object
    =retrieval>
        ISA         meaning
        word        =word
    =imaginal>
        ISA         question
        object      nil
 ==>
    =imaginal>
        object       =retrieval
    +visual-location>
      ISA            visual-location
      :attended      nil
 )

 (P encode-category
    =retrieval>
        ISA         meaning
        word        =word
    =imaginal>
        ISA         question
        object       =object
        category     nil
 ==>
    =imaginal>
        category        =retrieval
 )

 (p initial-retrieve
    =goal>
       ISA         is-member
    =imaginal>
       ISA         question
       object      =obj
       category    =cat
       judgment    nil
 ==>
    =imaginal>
       judgment    pending
    +retrieval>
       ISA         property
       object      =obj
       attribute   category
 )

(P direct-verify
   =goal>
      ISA         is-member
   =imaginal>
      ISA         question
      object      =obj
      category    =cat
      judgment    pending
   =retrieval>
      ISA         property
      object      =obj
      attribute   category
      value       =cat
   ?manual>
     state       free
==>
   =goal>
   +manual>
    ISA           press-key
    key           "j"
)

(P chain-category
   =goal>
      ISA         is-member
  =imaginal>
     ISA         question
     object      =obj1
     category    =cat
     judgment    pending
   =retrieval>
      ISA         property
      object      =obj1
      attribute   category
      value       =obj2
    - value       =cat
==>
   =goal>
   +retrieval>
      ISA         property
      object      =obj2
      attribute   category
    =imaginal>
       object      =obj2
)

(P mismatch1
   =goal>
      ISA         is-member
  =imaginal>
     ISA         question
     object      =obj
     category    =cat
     judgment    pending
   =retrieval>
      ISA         property
      - object    =obj
    ?manual>
       state      free
==>
    +manual>
     ISA           press-key
     key           "f"
)

(P mismatch2
   =goal>
      ISA         is-member
  =imaginal>
     ISA         question
     object      =obj
     category    =cat
     judgment    pending
   =retrieval>
      ISA         property
      - attribute    category
    ?manual>
       state      free
==>
    +manual>
     ISA           press-key
     key           "f"
)

 (P fail
   =goal>
      ISA         is-member
  =imaginal>
     ISA         question
     object      =obj
     category    =cat
     judgment    pending
   ?retrieval>
      state       error
   ?manual>
      state       free
==>
   =goal>
      judgment    no
   +manual>
       isa        press-key
       key        "f"
)


(goal-focus goal1)

(set-base-levels (shark 10) (dangerous 10) (locomotion 10) (swimming 10)
    (fish 10) (salmon 10) (edible 10) (breathe 10) (gills 10) (moves 10)
    (skin 10) (color 10) (sings 10) (ostrich 10) (flies 10) (height 10)
    (wings 10) (flying 10) (true 10) (false 10) (canary 10) (bird 10)
    (animal 10)))
