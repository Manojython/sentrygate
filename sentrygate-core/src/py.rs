//! PyO3 binding: `import sentrygate_core` gives the same Rust Scanner, GIL released during inference.

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::Scanner;

#[pyclass(name = "Scanner")]
struct PyScanner {
    inner: Scanner,
}

#[pymethods]
impl PyScanner {
    /// Scanner(tokenizer_json, onnx_path, questions=None, threshold=None)
    #[new]
    #[pyo3(signature = (tokenizer_json, onnx_path, questions=None, threshold=None))]
    fn new(
        tokenizer_json: &str,
        onnx_path: &str,
        questions: Option<Vec<String>>,
        threshold: Option<f32>,
    ) -> PyResult<Self> {
        let inner = match questions {
            Some(q) => Scanner::new(
                tokenizer_json,
                onnx_path,
                q,
                threshold.unwrap_or(crate::INJECTION_THRESHOLD),
            ),
            None => Scanner::injection(tokenizer_json, onnx_path),
        }
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Scan one text; returns a dict {is_attack, score, threshold, top_question, per_question, preview}.
    fn scan<'py>(&self, py: Python<'py>, text: &str) -> PyResult<Bound<'py, PyDict>> {
        // Release the GIL for the forward pass — this is the point of a Rust core.
        let finding = py
            .allow_threads(|| self.inner.scan(text))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let d = PyDict::new_bound(py);
        d.set_item("is_attack", finding.is_attack)?;
        d.set_item("score", finding.score)?;
        d.set_item("threshold", finding.threshold)?;
        d.set_item("top_question", finding.top_question)?;
        let per = PyDict::new_bound(py);
        for (q, s) in finding.per_question {
            per.set_item(q, s)?;
        }
        d.set_item("per_question", per)?;
        d.set_item("preview", finding.preview)?;
        Ok(d)
    }
}

#[pymodule]
fn sentrygate_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyScanner>()?;
    Ok(())
}
