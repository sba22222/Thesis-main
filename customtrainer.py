from transformers import Trainer
import wandb
import note_seq

from utils import token_sequence_to_note_sequence

# first create a custom trainer to log prediction distribution
SAMPLE_RATE=44100

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def evaluation_loop(
        self,
        dataloader,
        description,
        prediction_loss_only=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        # Call super class method to get the eval outputs
        eval_output = super().evaluation_loop(
            dataloader,
            description,
            prediction_loss_only,
            ignore_keys,
            metric_key_prefix,
        )

         # Log the prediction distribution using `wandb.Histogram` method.
        if wandb.run is not None:
            # Encode a starting token to begin the generation
            input_ids = self.tokenizer.encode("PIECE_START", return_tensors="pt").cuda()

            # Generate more tokens for each voice
            for voice_num in range(1, 5):
                generated_ids = self.model.generate(
                    input_ids,
                    max_length=512,
                    do_sample=True,
                    temperature=0.75, # Set temperature for sampling (higher values for more randomness, lower for more determinism)
                    #top_p = 0.9, # Set top-p sampling parameters (nucleus sampling) to control diversity
                    #top_k = 50, # Set top-k sampling parameters to restrict generation to the top-k most likely tokens
                    eos_token_id=self.tokenizer.encode("TRACK_END")[0]
                )

                # Decode the generated tokens into a token sequence
                token_sequence = self.tokenizer.decode(generated_ids[0])

                # Convert the token sequence into a NoteSequence
                note_sequence = token_sequence_to_note_sequence(token_sequence)

                # Synthesize the audio from the NoteSequence
                synth = note_seq.fluidsynth
                array_of_floats = synth(note_sequence, sample_rate=SAMPLE_RATE)

                # Convert the float audio samples to int16 format
                int16_data = note_seq.audio_io.float_samples_to_int16(array_of_floats)

                # Log the generated audio using the wandb.Audio method
                wandb.log({"Generated_audio_voice_" + str(voice_num): wandb.Audio(int16_data, SAMPLE_RATE)})

        # Return the evaluation output
        return eval_output