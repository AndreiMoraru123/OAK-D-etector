import blobconverter
import onnx
import torch
import argparse
from onnxsim import simplify


def deploy(model_path: str, new_model: str):
    """
    Deploy model to ONNX
    :param model_path: the path to the model checkpoint, such as 'checkpoints/checkpoint_ssd300.pt'
    :param new_model: the name of the ONNX model, such as 'ssd300'
    :return:
    """

    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location='cuda')
    model = checkpoint['model']
    model.eval()

    # Export to ONNX
    input_names = ['input']
    output_names = ['boxes', 'scores']
    dummy_input = torch.randn(1, 3, 300, 300).to('cuda')
    torch.onnx.export(model, dummy_input, new_model+'.onnx', verbose=True,
                      input_names=input_names, output_names=output_names)

    # Simplify ONNX
    simple_model, check = simplify('ssd300.onnx')
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(simple_model, new_model+'-sim'+'.onnx')


# Optional: Deploy model to Blob
def deploy_blob(model_name: str, output_dir: str):
    """
    Deploy model to Blob
    :param model_name: the name of the ONNX model, such as 'ssd300'
    :param output_dir: the path to the output directory, such as 'models'
    :return:
    """

    blobconverter.from_onnx(model=model_name+'-sim'+'.onnx',
                            output_dir=output_dir,
                            data_type='FP16',
                            use_cache=True,
                            shaves=6)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deploy model to OpenVINO')
    parser.add_argument('--model', type=str, default="checkpoints/checkpoint_ssd300.pt",
                        help='the path to the model checkpoint')
    parser.add_argument('--new_model', default="ssd300", type=str,
                        help='the name of the ONNX model')

    args = parser.parse_args()

    deploy(args.model, args.new_model)
    deploy_blob(args.new_model, 'models')
