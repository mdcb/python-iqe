#include "Python.h"
#include "structmember.h"

#include "numpy/arrayobject.h"

//extern "C" {
int iqe(float *pfm, float *pwm, int mx, int my, float *parm, float *sdev);
//}

#include "stdio.h"

static char const rcsid[] = "$Id:$";


// the object refering to this module

PyObject *pyiqe_module;
PyDoc_VAR(pyiqe_doc) = PyDoc_STR("Image Quality Estimator for Python.\n");

// the module common error

PyObject *pyiqe_error;

//=============================================================================
// Methods
//=============================================================================

PyDoc_STRVAR(pyiqe_iqe_doc,
"usage: iqe(2d-array-data[,2d-array-mask])\n\
returns [meanX,meanY,fwhmX,fwhmY,symetryAngle,objectPeak,meanBackground]\n\
where x,y = 0,0 is at the center of the first pixel.");

PyObject * pyiqe_iqe(PyObject *self, PyObject *args) {
   
   PyObject *data;
   PyObject *mask=NULL;
   npy_intp *dims,*mdims;
   npy_intp w,h,mw,mh;
   
   if (!PyArg_ParseTuple(args, "O|O", &data, &mask)) {
      PyErr_Format(pyiqe_error, "Usage: iqe(data[,mask])");
      return NULL;
   }

   // data
   // creates a new ref
   PyArrayObject *inp_array = (PyArrayObject *) PyArray_FROM_O(data);

   if (!inp_array) {
      PyErr_Format(pyiqe_error, "argument doesn't look like an array");
      return NULL;
   }

   if (PyArray_NDIM(inp_array) != 2) {
      PyErr_Format(pyiqe_error, "argument must be 2 dimensional");
      return NULL;
   }

   dims = PyArray_DIMS(inp_array);
   
   w=dims[1];
   h=dims[0];
   
   PyArrayObject *flt_array = (PyArrayObject *) PyArray_Cast(inp_array,NPY_FLOAT);
   if (!flt_array) {
      PyErr_Format(pyiqe_error, "argument doesn't cast to float");
      return NULL;
   }

   
   float *fltdata = (float*)PyArray_DATA(flt_array);
   
   // mask
   // creates a new ref
   PyArrayObject *msk_array=NULL;
   PyArrayObject *mskflt_array=NULL;
   float *fltmsk = NULL;
   if (mask) {
      msk_array = (PyArrayObject *) PyArray_FROM_O(mask);

      if (!msk_array) {
         PyErr_Format(pyiqe_error, "mask argument doesn't look like an array");
         return NULL;
      }

      if (PyArray_NDIM(msk_array) != 2) {
         PyErr_Format(pyiqe_error, "mask argument must be 2 dimensional");
         return NULL;
      }

      mdims = PyArray_DIMS(msk_array);
      mw=mdims[1];
      mh=mdims[0];

      mskflt_array = (PyArrayObject *) PyArray_Cast(msk_array,NPY_FLOAT);
      if (!mskflt_array) {
         PyErr_Format(pyiqe_error, "mask argument doesn't cast to float");
         return NULL;
      }
      fltmsk = (float*)PyArray_DATA(mskflt_array);
   }
   
#ifdef PY_IQE_DEBUG
   int i,j;
   float *pf=fltdata;
   printf ("data:\n");
   for (j=0;j<h;j++) {
      for (i=0;i<w;i++) {
         printf ("%.2f ",*pf++);
      }
      printf("\n");
   }
   pf=fltmsk;
   if (fltmsk) {
      printf ("mask:\n");
      for (j=0;j<h;j++) {
         for (i=0;i<w;i++) {
            printf ("%.2f ",*pf++);
         }
         printf("\n");
      }
   }
#endif
   
   float parm[8], sdev[8];
   
   int status = (iqe(fltdata, fltmsk, w, h, parm, sdev)!=0);
   
    if (status != 0) {
      // free temp ref
      // not needed? won't hurt anyway
      Py_DECREF(inp_array);
      if (msk_array) Py_DECREF(msk_array);
      // do
      Py_DECREF(flt_array);
      if (mskflt_array) Py_DECREF(mskflt_array);
      PyErr_Format(pyiqe_error, "Could not calculate statistics on specified area of image. Please make another selection.");
      return NULL;
    } else {
#ifdef PY_IQE_DEBUG
      printf ("meanX %f meanY %f fwhmX %f fwhmY %f\n",parm[0],parm[2],parm[1],parm[3]);
      printf ("symetryAngle %f objectPeak %f meanBackground %f\n",parm[4],parm[5],parm[6]);
#endif
    }
   
   npy_intp res_dims[1]={7};
   
   PyObject *result = PyArray_SimpleNewFromDescr(
      1,
      res_dims,
      PyArray_DescrFromType(NPY_FLOAT32));
   
   float *resdata = (float*)PyArray_DATA(result);
   resdata[0]=parm[0];
   resdata[1]=parm[2];
   resdata[2]=parm[1];
   resdata[3]=parm[3];
   resdata[4]=parm[4];
   resdata[5]=parm[5];
   resdata[6]=parm[6];
   
   
   // free temp ref - not sure it's really needed. won't hurt anyway
   Py_DECREF(inp_array);
   if (msk_array) Py_DECREF(msk_array);
   
   // this one for sure
   Py_DECREF(flt_array);
   if (mskflt_array) Py_DECREF(mskflt_array);
   
   //Py_INCREF(Py_None);
	//return Py_None;
   
   // returns the result as a numpy array
   Py_INCREF(result);
	return result;
}


// the module static method

static struct PyMethodDef pyiqe_methods[] = {
	   {"iqe",   (PyCFunction)pyiqe_iqe,   METH_VARARGS, pyiqe_iqe_doc},
	   {NULL}	/* sentinel */
	};

//-------------------------------------------------------------------
// initiqe - module initialization
//-------------------------------------------------------------------

PyMODINIT_FUNC initiqe() {

   // initialize Python
	Py_Initialize();
	
   // initialize numpy
   if (_import_array() < 0)
      return;

	// Initialize the module
	if ((pyiqe_module = Py_InitModule3("iqe",pyiqe_methods,pyiqe_doc)) == NULL)
      return;

	// define module generic error
	pyiqe_error = PyErr_NewException("iqe.error",NULL,NULL);
	PyModule_AddObject(pyiqe_module, "error" ,pyiqe_error);

	// Add symbolic constants
   PyModule_AddStringConstant(pyiqe_module,"version", PY_IQE_VERSION);


	}


