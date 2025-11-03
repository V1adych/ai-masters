import csv
import os
import argparse
import sys
import traceback
from contextlib import contextmanager
from torchmetrics.classification import BinaryF1Score


class TestDirectoryNotFoundError(FileNotFoundError):
    pass


def load_json_from_file(filename):
    with open(filename, "r") as f:
        return json.load(f, strict=False)


def test_linear_kernel(test_path, EPS=1e-9, SCALE=2, variant=None):
    x = np.linspace(-2, 2, num=5)
    y = np.abs(x)
    x_new = np.linspace(-2, 2, num=SCALE*x.shape[0])
    try:
        assert np.linalg.norm(interpolate(y, SCALE, linear_kernel) - interp1d(x, y, kind='linear', fill_value="extrapolate")(x_new)) < EPS
    except:
        return (0, f'RE: {traceback.format_exception(*sys.exc_info())}')
    try:
        original_length = 8
        x_original = np.linspace(0, 2 * np.pi, original_length)
        signal = np.sin(x_original)
        x_new = np.linspace(0, 2 * np.pi, int(original_length * SCALE))

        interpolated_signal_linear = interpolate(signal, SCALE, linear_kernel)

        with open(os.path.join(test_path, 'linear.bin'), 'rb') as f:
            target = np.frombuffer(f.read(), dtype=np.float64).reshape(int(original_length * SCALE))
            
        assert np.linalg.norm(interpolated_signal_linear - target) < EPS
    except:
        return (0.5, f'RE: {traceback.format_exception(*sys.exc_info())}')
    return (1, 'OK')

def test_cubic_kernel(test_path, EPS=1e-9, SCALE=2, variant=None):
    x = np.linspace(-2, 2, num=5)
    y = np.abs(x)
    x_new = np.linspace(-2, 2, num=SCALE*x.shape[0])
    try:
        assert np.linalg.norm(
            interpolate(y, SCALE, cubic_kernel) - np.array(
                [2.0, 1.648267009, 1.121418827, 0.5925925925, 0.0877914952, 0.0877914952, 0.5925925925, 1.121418827, 1.648267009, 2.0]
            )
        ) < EPS
    except:
        return (0, f'RE: {traceback.format_exception(*sys.exc_info())}')
    try:
        original_length = 8
        x_original = np.linspace(0, 2 * np.pi, original_length)
        signal = np.sin(x_original)
        x_new = np.linspace(0, 2 * np.pi, int(original_length * SCALE))

        interpolated_signal_cubic = interpolate(signal, SCALE, cubic_kernel)

        with open(os.path.join(test_path, 'cubic.bin'), 'rb') as f:
            target = np.frombuffer(f.read(), dtype=np.float64).reshape(int(original_length * SCALE))
            
        assert np.linalg.norm(interpolated_signal_cubic - target) < EPS
    except:
        return (0.5, f'RE: {traceback.format_exception(*sys.exc_info())}')
    return (1, 'OK')

def test_swinir(test_path, EPS=10, SCALE=4, variant=None):
    if variant is None:
        swinir = cv2.imread('swinir_board.png')[..., ::-1]
    else:
        swinir = cv2.imread('swinir_zaya.png')[..., ::-1]
    with open(os.path.join(test_path, 'swinir.bin'), 'rb') as f:
        target = np.frombuffer(f.read(), dtype=np.uint8).reshape(swinir.shape)
        
    assert l2_norm(swinir, target) < EPS

    return (2, 'OK')

def evaluate_model(model, test_loader, device):
    model.eval()
    total_iou = 0
    num_batches = 0

    f1_metric = BinaryF1Score(threshold=model.threshold).to(device)
    f1_metric.reset()

    with torch.no_grad():
        for batch in test_loader:
            for key in batch.keys():
                batch[key] = batch[key].to(device)
            img, mask, gt, has_art = batch['img'], batch['mask'], batch['gt'], batch['has_artifact'].float()

            model_input = {'img': img, 'gt': gt}
            pred_mask = model(model_input)
            bin_mask = (pred_mask >= model.threshold).float()
            pred_has_art = (bin_mask.sum(dim=(1, 2, 3)) > 0).float()

            f1_metric.update(pred_has_art, has_art)
            total_iou += iou(bin_mask, mask)

            num_batches += 1

    f1_score = f1_metric.compute().item()
    avg_iou = (total_iou / num_batches).item()
    
    return round(avg_iou, 4), round(f1_score, 4)


def test_model(test_path, variant=None):
    test_dataset_dir = test_path + '/test_dataset/'
    test_labels_path = test_path + '/labels.csv'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    _, test_loader = create_dataloader(test_dataset_dir, test_labels_path, batch_size=8, val_size=1, random_state=42)

    model = MyModel().to(device)
    model.load_weights('my_model.pth', device)
    avg_iou, fscore = evaluate_model(model, test_loader, device)

    return (avg_iou, fscore, 'OK')


def run_tests(test_path):
    test_info = load_json_from_file(os.path.join(test_path, 'info.json'))
    if len(test_info) > 1:
        test_name, variant = test_info
    else:
        test_name, variant = test_info[0], None
    
    print(test_name)
    tests = {
        'linear_kernel': test_linear_kernel,
        'cubic_kernel': test_cubic_kernel,
        'swinir': test_swinir,
        'model': test_model,
    }
    results = tests[test_name](test_path, variant=variant)
    results = tuple((results, )) if len(results) == 2 else tuple(((results[0], f'avg_iou/{results[2]}'), (results[1], f'fscore/{results[2]}')))
    return results


def save_results(results, filename):
    header = ['score', 'conclusion']
    with open(filename, 'w', newline='') as resfile:
        writer = csv.writer(resfile)
        writer.writerow(header)
        resfile.flush()
        for row in results:
            writer.writerow(row)
            resfile.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing script')
    parser.add_argument('--test_dir', type=str, default='test_files')
    parser.add_argument('--output_file', type=str, default='results.csv')

    args = parser.parse_args()

    try:
        results = run_tests(args.test_dir)
    except TestDirectoryNotFoundError as e:
        print('Failed to find test directory')
        print(e)
        sys.exit()

    save_results(results, args.output_file)
    print('Results saved to: {}'.format(args.output_file))

