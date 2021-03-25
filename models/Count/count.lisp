;; (load "/home/dfish/.julia/dev/ACTRTutorial/actr6/load-act-r-6.lisp")
;; (load "/home/dfish/.julia/dev/FundamentalToolsACTR/models/Count/count.lisp")
;; (generate-recovery-data 100 100 "/home/dfish/.julia/dev/ACTRTutorial/Tutorial_Models/Markov Chain Models/Semantic MP/recovery_data")

(clear-all)

(defvar *response*)
(defvar *response-time*)

;; num1 and num2 must be strings
(defun run-count-model (start end)
  (let ((window (open-exp-window "Count Model"
                                 :visible nil
                                 :width 600
                                 :height 300))
        (x 25))

    (reset)
    (install-device window)

    (dolist (text (list start  end))
      (add-text-to-exp-window :text text :x x :y 150 :width 75)
      (incf x 75))

    (setf *response* nil)
    (setf *response-time* nil)

    (proc-display)

    (run 30)
    (if (string-equal *response* "j")
        (setf data (/ *response-time* 1000.0))
        (setf data nil))))

(defmethod rpm-window-key-event-handler ((win rpm-window) key)
  (setf *response-time* (get-time t))

(setf *response* (string key)))

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

(defun run-n-times (n start end)
   (let ((rts (list)))
   (dotimes (i n)
      (push (list (run-count-model start end)) rts))
      rts))

(defun generate-data (n path)
    (let ((rts))
        (sgp :seed (719254 98))
        (setf rts (run-n-times n "2" "4"))
        (setf file-name (concatenate 'string path "/Lisp_Data.csv"))
        (write-to-file rts file-name)))

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



(define-model count

(sgp :esc t :blc 1.5 :ans .3 :rt -10 :v t :act t :trace-detail high :randomize-time t :VPFT t :VIDT t)

(chunk-type count-order current next)
(chunk-type count-from start end count)
(chunk-type number value name)
(chunk-type question start end)

(add-dm
    (zero ISA number value "0" name "zero")
    (one ISA number value "1" name "one")
    (two ISA number value "2" name "two")
    (three ISA number value "3" name "three")
    (four ISA number value "4" name "four")
    (five ISA number value "5" name "five")
    (six ISA number value "6" name "six")
    (seven ISA number value "7" name "seven")
    (eight ISA number value "8" name "eight")
    (nine ISA number value "9" name "nine")
    (a ISA count-order current "0" next "1")
    (b ISA count-order current "1" next "2")
    (c ISA count-order current "2" next "3")
    (d ISA count-order current "3" next "4")
    (e ISA count-order current "4" next "5")
    (f ISA count-order current "5" next "6")
    (g ISA count-order current "6" next "7")
    (h ISA count-order current "7" next "8")
    (i ISA count-order current "8" next "9")
    (current-goal ISA count-from)
 )

 (P start
     =goal>
        ISA         count-from
    ?imaginal>
        state       free
        buffer      empty
    ?visual-location>
        buffer      unrequested
    ==>
    +imaginal>
        ISA         question
    +visual-location>
        ISA         visual-location
        screen-x    lowest
 )

 (P attend-visual-location
     =imaginal>
         ISA         question
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
    =imaginal>
 )

 (P retrieve-meaning
     =visual>
        ISA         text
        value       =val
    ==>
     +retrieval>
         ISA        number
         value      =val
 )

 (P encode-start-number
    =retrieval>
        ISA        number
        value      =val
    =imaginal>
        ISA         question
        start      nil
 ==>
    =imaginal>
        start       =val
    +visual-location>
      ISA            visual-location
      :attended      nil
 )

 (P encode-end-number
     =goal>
        ISA         count-from
    =retrieval>
        ISA         number
        value       =val
    =imaginal>
        ISA         question
        start       =start
        end         nil
 ==>
    =imaginal>
        end        =val
    =goal>
        start      =start
        end        =val
 )

(p start-count
   =goal>
      ISA         count-from
      start       =num1
      count       nil
    ?retrieval>
      state free
 ==>
   =goal>
      count       =num1
   +retrieval>
      ISA         count-order
      current      =num1
      ;!output!       (=num1)
)

(P increment
   =goal>
      ISA         count-from
      count       =num1
    - end         =num1
   =retrieval>
      ISA         count-order
      current       =num1
      next          =num2
    ?retrieval>
      state free
 ==>
   =goal>
      count       =num2
   +retrieval>
      ISA         count-order
      current       =num2
   ;!output!       (=num1)
)

(P stop
   =goal>
      ISA         count-from
      count       =num
      end         =num
   ?manual>
     state free
 ==>
   -goal>
   ;!output!       (=num)
   +manual>
        ISA           press-key
        key           "j"

)

(set-base-levels
  (zero 10) (one 10) (two 10) (three 10) (four 10) (five 10)
  (six 10) (seven 10) (eight 10) (nine 10))

(goal-focus current-goal)
)
