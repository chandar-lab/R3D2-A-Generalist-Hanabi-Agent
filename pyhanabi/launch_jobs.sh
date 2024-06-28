for s in "a" "b" "d"; #
  do for l in "/home/mila/n/nekoeiha/scratch/llm_hanabi_hive/exps/test_drrn_b64_update_test_np_3_text_enc_frq_1_text_enc_pretrained_layers_1_s_88451596/epoch100.pthw" "/home/mila/n/nekoeiha/scratch/llm_hanabi_hive/exps/test_drrn_b64_update_test_np_3_text_enc_frq_1_text_enc_pretrained_layers_1_s_88451596/epoch200.pthw" "/home/mila/n/nekoeiha/scratch/llm_hanabi_hive/exps/test_drrn_b64_update_test_np_3_text_enc_frq_1_text_enc_pretrained_layers_1_s_88451596/epoch300.pthw" "/home/mila/n/nekoeiha/scratch/llm_hanabi_hive/exps/test_drrn_b64_update_test_np_3_text_enc_frq_1_text_enc_pretrained_layers_1_s_88451596/epoch500.pthw" "/home/mila/n/nekoeiha/scratch/llm_hanabi_hive/exps/test_drrn_b64_update_test_np_3_text_enc_frq_1_text_enc_pretrained_layers_1_s_88451596/epoch400.pthw" "/home/mila/n/nekoeiha/scratch/llm_hanabi_hive/exps/test_drrn_b64_update_test_np_3_text_enc_frq_1_text_enc_pretrained_layers_1_s_88451596/epoch600.pthw";
    do for o in "pretrained";
      do for p in 2 5;
        do for q in 1;
          do
            sbatch submit_transfer_job.sh $l $s $o $p $q;
          done;
        done;
      done;
    done;
  done