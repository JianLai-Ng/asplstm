python asplstm.py --inS 14 --outS 14 --y BOX_TEU --gran day —notTrial 1

--inS 14 (to pass n_steps_in as 14 time units)
--outS 14 (to pass n_steps_out as 14 time units)
--y1 BOX_TEU (to pass y variables as sums of BOX and TEU at time units)
--gran day(to pass day as time unit)
--notTrial 1 (1 if run actual grid search, 0 if run trial)