#include <stdio.h>
#include <Python.h>
#include <structmember.h>
#include <numpy/arrayobject.h>

extern int iqe(float * pfm, float * pwm, int mx, int my, float * parm, float * sdev);

PyObject * pyiqe_err;

#define PYIQE_ERR(m) PyErr_Format(pyiqe_err, m);

PyObject * pyiqe(PyObject * self, PyObject * args)
{

  PyObject * data = NULL;
  PyObject * mask = NULL;
  PyArrayObject * inp_array = NULL;
  PyArrayObject * flt_array = NULL;
  PyArrayObject * msk_array = NULL;
  PyArrayObject * mskflt_array = NULL;
  PyArrayObject * result = NULL;

  if (!PyArg_ParseTuple(args, "O|O", &data, &mask))
    {
      PYIQE_ERR("Usage: iqe(data[,mask])");
      goto iqe_exit;
    }

  inp_array = (PyArrayObject *) PyArray_FROM_O(data);

  if (!inp_array)
    {
      PYIQE_ERR("cannot convert input to array");
      goto iqe_exit;
    }

  if (PyArray_NDIM(inp_array) != 2)
    {
      PYIQE_ERR("input must be two dimensional");
      goto iqe_exit;
    }

  npy_intp * dims = PyArray_DIMS(inp_array);
  npy_intp w = dims[1]; // XXX subwindow -> param
  npy_intp h = dims[0];

  flt_array = (PyArrayObject *) PyArray_Cast(inp_array, NPY_FLOAT);

  if (!flt_array)
    {
      PYIQE_ERR("input does not cast to float");
      goto iqe_exit;
    }


  float * fltdata = (float *) PyArray_DATA(flt_array);
  float * fltmsk = NULL;

  if (mask)
    {
      msk_array = (PyArrayObject *) PyArray_FROM_O(mask);

      if (!msk_array)
        {
          PYIQE_ERR("cannot convert mask to array");
          goto iqe_exit;
        }

      if (PyArray_NDIM(msk_array) != 2)
        {
          PYIQE_ERR("mask must be two dimensional");
          goto iqe_exit;
        }

      npy_intp * mdims = PyArray_DIMS(msk_array);
      npy_intp mw = mdims[1];
      npy_intp mh = mdims[0];

      mskflt_array = (PyArrayObject *) PyArray_Cast(msk_array, NPY_FLOAT);

      if (!mskflt_array)
        {
          PYIQE_ERR("mask does not cast to float");
          goto iqe_exit;
        }

      fltmsk = (float *) PyArray_DATA(mskflt_array);
    }

  float parm[8], sdev[8];

  if (iqe(fltdata, fltmsk, w, h, parm, sdev))
    {
      Py_DECREF(inp_array);

      if (msk_array) { Py_DECREF(msk_array); }

      Py_DECREF(flt_array);

      if (mskflt_array) { Py_DECREF(mskflt_array); }

      PYIQE_ERR("Could not calculate statistics on specified area of image.");
      goto iqe_exit;
    }

  npy_intp res_dims[1] = {7};
  result = PyArray_SimpleNewFromDescr(1, res_dims, PyArray_DescrFromType(NPY_FLOAT32));

  float * rval = (float *) PyArray_DATA(result);
  rval[0] = parm[0];
  rval[1] = parm[2];
  rval[2] = parm[1];
  rval[3] = parm[3];
  rval[4] = parm[4];
  rval[5] = parm[5];
  rval[6] = parm[6];

iqe_exit:

  Py_XDECREF(inp_array);
  Py_XDECREF(flt_array);
  Py_XDECREF(msk_array);
  Py_XDECREF(mskflt_array);
  Py_XINCREF(result);
  return result;
}


//-------------------------------------------------------------------
// init
//-------------------------------------------------------------------

PyDoc_STRVAR(pyiqe_doc,
  "usage: iqe(2d-array-data[,2d-array-mask])\n"
  "returns [meanX,meanY,fwhmX,fwhmY,symetryAngle,objectPeak,meanBackground]\n"
  "where x,y = 0,0 is at the center of the first pixel."
);

struct PyMethodDef methods[] =
{
  {"iqe", (PyCFunction)pyiqe, METH_VARARGS, pyiqe_doc},
  {NULL}
};

struct PyModuleDef pyiqe_def =
{
  PyModuleDef_HEAD_INIT,
  "iqe",
  PyDoc_STR("Image quality estimator for python.\n"),
  -1,
  methods,
  NULL,
  NULL,
  NULL,
  NULL
};

PyMODINIT_FUNC PyInit_iqe(void)
{
  PyObject * m;
  Py_Initialize();
  import_array();
  m = PyModule_Create(&pyiqe_def);

  if (m == NULL) { return NULL; }

  pyiqe_err = PyErr_NewException("iqe.error", NULL, NULL);
  PyModule_AddObject(m, "error" , pyiqe_err);
  PyModule_AddStringConstant(m, "__version__", PYIQE_VERSION);
  return m;
}


