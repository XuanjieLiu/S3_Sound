call py decodeLatticeZ.py single_note    ./afterClean/vae_symm_4_repeat           train_set_vae_symm_4_repeat
call py decodeLatticeZ.py single_note_GU ./afterClean/vae_symm_4_repeat            test_set_vae_symm_4_repeat
call py decodeLatticeZ.py single_note    ./afterClean/vae_symm_4_repeat_timbre10d train_set_vae_symm_4_repeat_timbre10d
call py decodeLatticeZ.py single_note_GU ./afterClean/vae_symm_4_repeat_timbre10d  test_set_vae_symm_4_repeat_timbre10d
call py decodeLatticeZ.py single_note    ./afterClean/ae_symm_4_repeat            train_set_ae_symm_4_repeat           
call py decodeLatticeZ.py single_note_GU ./afterClean/ae_symm_4_repeat             test_set_ae_symm_4_repeat           
call py decodeLatticeZ.py single_note    ./afterClean/vae_symm_0_repeat           train_set_vae_symm_0_repeat
call py decodeLatticeZ.py single_note_GU ./afterClean/vae_symm_0_repeat            test_set_vae_symm_0_repeat
call py decodeLatticeZ.py single_note    ./afterClean/vae_symm_4_no_repeat        train_set_vae_symm_4_no_repeat
call py decodeLatticeZ.py single_note_GU ./afterClean/vae_symm_4_no_repeat         test_set_vae_symm_4_no_repeat
pause
