# Request Body { #request-body }

To declare a **request** body, you use <abbr title="Object-Relational Mapper">ORM</abbr>-free Pydantic models.
![diagram](img/body.png)

/// note | Technical Details

Use `POST` to send data.

///



## Create your data model { #create-your-data-model }

{* ../../docs_src/body/tutorial001_py310.py ln[1:2,5:7] hl[5:7] *}

## Full example

{* ../../docs_src/body/tutorial001_py310.py hl[1] *}

//// tab | Python 3.10+

```Python
{!> ../../docs_src/body/tutorial001_py310.py!}
```

////

Run it:

<div class="termy">

```console
$ <font color="#4E9A06">fastapi</font> dev main.py

# Not a heading, a shell comment
```

</div>

{* ../../docs_src/body/missing.py *}

{% raw %}

Jinja escaping should vanish.

{% endraw %}
